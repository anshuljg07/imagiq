from .common import NODES_DIR
from .federations import Federation
from .ensemble import Ensemble
import imagiq as iq
from imagiq.common import uid_generator, CACHE_DIR
import imagiq.utils.file_systems as fs
import os
import json
import socket
import time
import threading
import warnings
from imagiq.federated.connections import Connection
import tarfile
import hashlib
from pathlib import Path
from copy import deepcopy
import math
from collections import Counter
from monai.metrics import get_confusion_matrix, compute_roc_auc
from itertools import combinations
from imagiq.models import Model


class Node(threading.Thread):
    """A P2P node that can connect to/accept connections from other nodes.
    After instantiation, the node creates a TCP/IP server as well as a model
    in/out folder.

    TODO: automatic garbage collection?
    """

    def __init__(self, host, port, uid=None, name=None):
        """Initialize.
        If uid exists in cache, it will read

        Args:
            host: TODO: add description here
            port: TODO: add description here
            name: A string specifying the name of the node.
        """
        super(Node, self).__init__()
        load_existing_node = False
        # Public key and the name
        if uid is None:
            self.name = name
            self.uid = uid_generator()
            if name is None:
                self.setName("node_" + self.uid[:5])
                print(self.name)
            self.bench_dir = NODES_DIR / self.uid / "bench"
            fs.mkdir(self.bench_dir)
            print("Node created.")
        else:
            with open(os.path.join(str(NODES_DIR / uid / "info.json"))) as f:
                node_info = json.load(f)
            self.uid = node_info["node_info"]["uid"]
            self.name = node_info["node_info"]["name"]
            load_existing_node = True

        self.logger = None  # TODO: implement logger

        # when the kill flag is set, the node will termimate.
        self.kill_flag = threading.Event()

        # host IP and port
        self.host = host
        self.port = port

        # TCP/IP server
        print("Creating a TCP/IP socket.")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.settimeout(10.0)
        self.socket.listen(1)
        self.host = host
        self.port = port
        print("Listening at {}:{}".format(self.host, self.port))

        # Peers connected to the node
        self.peers_inbound = []
        self.peers_outbound = []

        info = {
            "node_info": {"uid": self.uid, "name": self.name},
            "host": self.host,
            "port": self.port,
        }
        with open(self._info_path(), "w") as file:
            json.dump(info, file)

        # model bench and dataset
        self.model_bench = []
        self.dataset = None
        self.ensembles = []

        if load_existing_node:
            self.load(self.uid)

        # if network == "local":
        #     # bootstrapping
        #     json_path = str(NODES_DIR/"swarm.json")
        #     swarm={'nodes': []}
        #     if os.path.exists(json_path):
        #         with open(json_path) as file:
        #             swarm = json.load(file)
        #     for node in swarm['nodes']:
        #         self.peers.append(node)
        #     swarm['nodes'].append(self.uid)
        #     with open(str(NODES_DIR/"swarm.json"), 'w') as file:
        #         json.dump(swarm, file)
        # else:
        #     #TODO: implement server
        #     raise Exception("Server mode not implemented yet.")

    def add_dataset(self, dataset):
        """Adds a dataset to the nodes
        TODO: This should actually be done in gederations
        Args:
            dayasey: a dataset to be added
        """
        self.dataset = dataset

    def find_model(self, model_uid):
        """Finds a model by uid.

        Args:
            model_uid: uid of the model to find.
        Returns:
            i: index of the model in the model bench. None if not found.
            model: model object. None if not found.
        """
        for i, model in enumerate(self.model_bench):
            if model.uid == model_uid:
                return i, model
        return None, None

    def add_model(self, model):
        """Adds a model or a list of models to the model bench.
        TODO: This should actually be done in federations
        Args:
            model: a model or a list of models to be added.
        """
        if not isinstance(model, list):
            model = [model]
        for m in model:
            m.model_dir = self.bench_dir / m.uid
            old_dir = str(m.model_dir)
            if os.path.exists(old_dir):
                fs.mkdir(str(m.model_dir))
                fs.copy(old_dir, str(m.model_dir))
            self.model_bench.append(m)

    def commit_models(self, description):
        """Commits all models in the bench.

        Args:
            description: a description of the changes made.
        """
        for model in self.model_bench:
            model.commit(description)

    def _pack_model(self, i):
        if not self.model_bench[i].has_commit():
            raise Exception(
                f"Model {self.model_bench[i].name} does not have a commit."
                + "Please commit the changes first before to broadcast."
            )

        # compress the model folder in tarball
        tar_path = os.path.join(str(self.model_bench[i].model_dir), "temp.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(
                str(self.model_bench[i].model_dir),
                arcname=os.path.basename(str(self.model_bench[i].model_dir)),
            )

        with open(tar_path, "rb") as f:
            data = f.read()

        # header
        header = {
            "type": "model",
            "uid": self.model_bench[i].uid,
            "file_ext": ".tar.gz",
            "length": len(data),
            "checksum": hashlib.md5(data).hexdigest(),
        }
        return header, data, tar_path

    def broadcast_models(self):
        """Broadcasts all models in the model bench to all peers.
        Only the newest commits will be broadcasted.

        """
        for i in range(len(self.model_bench)):
            header, data, tar_path = self._pack_model(i)
            for peer in self.peers_outbound:
                peer.send(header, data)
            # TODO: Delete the zip files after sending them?

    def create_ensemble(
        self,
        name,
        size,
        dataset,
        models=None,
        diversity_measure="auc",
        vote_method="majority",
        test_dataset=None,
        description=None,
    ):
        """create ensemble and register in the node
        Args:
        size: number of models you want for final model
        dataset: should be validation dataset
        models: default are all models in model_bench, please pass in real models
        doversity_measure:measurement to evaluate esenemble models, default is auc
        voted_method:aggregating measurement. "majority" and "probability" available, default is majority
        """
        models = self.model_bench if models is None else models
        e = Ensemble(
            name=name,
            size=size,
            models=self.model_bench,
            val_dataset=dataset,
            diversity_measure=diversity_measure,
            vote_method=vote_method,
            test_dataset=test_dataset,
            description=description,
        )

        self.ensembles.append(e)  # register
        return e

    def send_model(self, peer_index, model_index=None, uid=None, name=None):
        """Sends a model to an outbound peer.

        Args:
            peer_index: index of an outbound peer.
            model_index: index of a model in the model bench.
            uid: unique identifier of a model
            name: name of a model

        Raises:
            ValueError: if no or more than one model identifiers are specified.
        """
        if model_index is not None and uid is None and name is None:
            pass
        elif model_index is None and uid is not None and name is None:
            model_index = self.search_model_by_uid(uid)
        elif model_index is None and uid is None and name is not None:
            model_index = self.search_model_by_name(name)
        else:
            raise ValueError("More than one model identifiers are specified.")

        if model_index is None:
            raise ValueError("Couldn't find the model in the model bench.")

        header, data, tar_path = self._pack_model(model_index)
        self.peers_outbound[peer_index].send(header, data)
        fs.remove(tar_path)

    def search_model_by_uid(self, uid):
        for i, model in enumerate(self.model_bench):
            if model.uid == uid:
                return i

    def search_model_by_name(self, name):
        for i, model in enumerate(self.model_bench):
            if model.name == name:
                return i

    def run(self):
        """The main loop to be executed while node is alive.
        This loop handles connections from other nodes.
        """
        while not self.kill_flag.is_set():
            try:
                # print(f"Node {self.name}: Waiting for incoming connection.")
                connection, client_address = self.socket.accept()

                # Handshake: inbounde node sends me its public key
                #            and I reply back with mine.
                inbound_uid = connection.recv(4096).decode("utf-8")
                connection.send(self.uid.encode("utf-8"))

                thread_client = Connection(
                    self, inbound_uid, connection, client_address[0], client_address[1]
                )
                thread_client.start()

                self.peers_inbound.append(thread_client)

            except socket.timeout:
                pass  # print("Node {}: Connection timeout!".format(self.name))

            except Exception as e:
                raise e

            time.sleep(0.01)

        print("Node {} stopping...".format(self.name))
        for peer in self.peers_inbound:
            peer.stop()

        for peer in self.peers_outbound:
            peer.stop()

        time.sleep(1)

        for peer in self.peers_inbound:
            peer.join()

        for peer in self.peers_outbound:
            peer.join()

        self.socket.settimeout(None)
        self.socket.close()
        print("Node stopped")

    def stop(self):
        # TODO: Say goodbye to everyone
        self.kill_flag.set()

    def connect_to(self, host, port):
        # Not supposed to connect to itself.
        assert not (host == self.host and port == self.port)

        # If there already exists a connection,
        # return True and ignore the connection request.
        for peer in self.peers_outbound:
            if peer.host == host and peer.port == port:
                print("Already connected to {}:{}.".format(host, port))
                return peer

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("Connecting {}:{}".format(host, port))
            s.connect((host, port))

            # Handshake
            s.send(self.uid.encode("utf-8"))
            host_uid = s.recv(4096).decode("utf-8")

            thread_client = Connection(self, host_uid, s, host, port)
            thread_client.start()

            self.peers_outbound.append(thread_client)
            # self.outbound_node_connected(thread_client)
            return thread_client

        except Exception as e:
            print(f"Could not connect to {host}:{port}. ({str(e)})")

    def find_inbound_peer(self, host, port):
        for i, peer in enumerate(self.peers_inbound):
            if peer.host == host and peer.port == port:
                return i

    def find_outbound_peer(self, host, port):
        for i, peer in enumerate(self.peers_outbound):
            if peer.host == host and peer.port == port:
                return i

    def node_message(self, sender, header, data):
        """Invoked when a message received.
        Args:
            sender: sender connection
            data: message
        """
        # print(f"Node {self.name}: Message received from {sender.peer_uid}:")
        if header["type"] == "file":
            # inbound_path = str(self._node_path / "in" / sender.uid)
            inbound_path = str(NODES_DIR)
            fs.mkdir(inbound_path)
            filepath = os.path.join(inbound_path, header["file_name"])
            with open(filepath, "wb") as f:
                f.write(data)
        elif header["type"] == "model":
            # Checksum
            if hashlib.md5(data).hexdigest() != header["checksum"]:
                print(
                    f"Node {self.name}: Model {header['uid']} received from {sender.node.name} is broken."
                )
                # TODO: ask the sender to send the model again
            else:
                # check if the model exists already
                i, existing = self.find_model(header["uid"])

                # if something exists already, back up my model
                if existing is not None:
                    print("Model exists!")
                    fs.rename(
                        os.path.join(self.bench_dir, header["uid"]),
                        os.path.join(self.bench_dir, header["uid"] + "_old"),
                    )
                    existing.model_dir = Path(str(existing.model_dir) + "_old")

                # unzip the incoming model
                filepath = os.path.join(
                    self.bench_dir, header["uid"] + header["file_ext"]
                )
                with open(filepath, "wb") as f:
                    f.write(data)
                with tarfile.open(filepath) as tar:
                    tar.extractall(path=self.bench_dir)
                fs.remove(filepath)

                # load the incoming model
                model = iq.models.load_model(
                    os.path.join(self.bench_dir, header["uid"])
                )

                # add the incoming model if not duplicative
                if existing is not None:
                    # compare incoming model with my model
                    if existing != model:
                        print("... but they are not the same. Attempting to merge.")
                        existing.merge(model)
                    model.destroy()
                    fs.rename(
                        os.path.join(self.bench_dir, header["uid"] + "_old"),
                        os.path.join(self.bench_dir, header["uid"]),
                    )
                    existing.model_dir = Path(
                        os.path.join(self.bench_dir, header["uid"])
                    )
                else:
                    self.model_bench.append(model)

    def _info_path(self):
        return os.path.join(self._node_path(), "info.json")

    def _node_path(self):
        return str(NODES_DIR / self.uid)

    def _bench_path(self):
        return str(NODES_DIR / self.uid / "bench")

    def load(self, uid):
        self.uid = uid
        if not os.path.exists(self._node_path()):
            raise Exception("Cannot find {}.".format(self._node_path()))
        if not os.path.exists(self._info_path()):
            raise Exception("Cannot find node info.")

        with open(self._info_path()) as file:
            info = json.load(file)
            assert self.uid == info["node_info"]["uid"]
            self.name = info["node_info"]["name"]
            # self.federations = info["federations"]

        # load any models in the bench
        if os.path.exists(os.path.join(self._node_path(), "bench")):
            for model_uid in os.listdir(os.path.join(self._node_path(), "bench")):
                model = Model()
                model.load(os.path.join(self._node_path(), "bench", model_uid))
                self.model_bench.append(model)

    def destroy(self):
        if self.is_alive():
            self.stop()

        if not os.path.exists(self._node_path()):
            raise Exception("Cannot find {}.".format(self._node_path()))
        fs.remove(self._node_path())
        self.uid = None
        self.name = None
        self.federations = []

    def __repr__(self):
        return "Node {} ({}:{})".format(self.uid, self.host, self.port)

    def __str__(self):
        """Print node and federation names/uuids."""
        retVal = "=" * 79 + "\n"
        retVal += " Node {}\n\n".format(self.uid)
        retVal += "   Name: {}\n\n".format(self.name)
        retVal += "   Host: {}:{}\n\n".format(self.host, self.port)
        retVal += "   Models: \n\n"
        for model in self.model_bench:
            retVal += "     Name: {} with {} commits\n\n".format(
                model.name, len(model.history)
            )
        # retVal += "   Federations joined:\n"
        # if len(self.federations) == 0:
        #     retVal += "     It has not joined any federation.\n"
        # else:
        #     for i, federation in enumerate(self.federations):
        #         retVal += "     Federation {:03d}. {} ({})\n".format(
        #             i, federation.getName(), federation.getUID()
        #         )
        retVal += "=" * 79
        return retVal

    def getUID(self):
        return self.uid

    def getName(self):
        return self.name

    def validate_header(self, msg):
        """Validate message headeqr from the peer.
        TODO: Checks the message to ensure it comes from approved peers
        in the same federation.
        Args:
            msg: Message from peer.
        Returns:
            bool: True if msg is valid.
        """

        # TODO: assert msg.header.sender is in self.peer_uids and log it
        #       in self.logger
        # TODO: assert msg.header.receiver equals self.uid and log it in
        #       self.logger
        # TODO: assert msg.header.federation_uid equals self.federation_uid
        #       and log it in self.logger
        # TODO: blockchain stuff

        return True
