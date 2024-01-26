import socket
import threading
import json
import tqdm
import psutil
from random import randint


class Connection(threading.Thread):
    """TCP/IP socket connection between two peer nodes.

    All TCP/IP communications are handled by this class.
    """

    def __init__(self, node, peer_uid, sock, host, port):
        """Instantiates a new connection.

        node: Node receiving the connection.
        peer_uid: Node requesting the connection.
        sock: Socket associated with the connection.
        host: IP address.
        port: Port number.
        """

        super(Connection, self).__init__()

        self.node = node
        self.peer_uid = peer_uid
        self.host = host
        self.port = port
        self.socket = sock
        self.kill_flag = threading.Event()

        self.info = {}

        print(
            "Node {}: Connection started with {} at {}:{}.".format(
                self.node.name, self.peer_uid, self.host, self.port
            )
        )

    def run(self):
        """The main loop of the thread to handle the connection with the node.
        Within the main loop the thread waits to receive data from the node.
        If data is received the method node_message will be invoked of the
        main node to be processed."""
        self.socket.settimeout(10.0)
        buffer = b""  # Hold the stream that comes in!
        packet_size = 0
        progress = None

        while not self.kill_flag.is_set():
            chunk = b""

            try:
                chunk = self.socket.recv(32768)

            except socket.timeout:
                pass  # print("Connection: timeout")

            except Exception as e:
                self.kill_flag.set()
                print("Unexpected error (" + e + ").")

            # BUG: when expected packet size is non-zero,
            #  but nothing is incoming
            if chunk != b"":
                buffer += chunk
                if progress is not None:
                    progress.update(len(chunk))

            if len(buffer) > 4 and packet_size == 0:
                packet_size = int.from_bytes(buffer[:4], "big")
                # print("Received:", packet_size)
                buffer = buffer[4:]
                # TODO: Progress Bar
                progress = tqdm.tqdm(
                    range(packet_size),
                    f"{self.node.name} is receiving data",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
                progress.update(len(buffer))

            if len(buffer) >= packet_size and packet_size != 0:
                progress.close()
                packet = buffer[:packet_size]
                buffer = buffer[packet_size:]

                header, data = self.parse_packet(packet)
                self.node.node_message(self, header, data)

                packet_size = 0
                progress = None

            # time.sleep(0.005)

        # IDEA: Invoke (event) a method in main_node so the user is able to
        # send a bye message to the node before it is closed?

        self.socket.settimeout(None)
        self.socket.close()
        print("Connection: Stopped")

    def parse_packet(self, packet):
        """Parses the packet.

        First 4 bytes of each packet is designed to be an integer for the
        header size. The following bytes are the header and the actual meat
        of the data.

        Args:
            packet: A packet.
        returns:
            header: A dictionary (JSON) header.
            data: Data. Either string, dict, or binary.
        """
        # Parse the header
        # first 4 bytes are always for the header size
        header_length = int.from_bytes(packet[0:4], "big")
        header = json.loads(packet[4 : header_length + 4].decode("utf-8"))

        # parse the actual meat of data
        data = packet[header_length + 4 :]
        try:  # try if data can be decoded into a string
            data_decoded = data.decode("utf-8")
            try:  # try if data can be decoded into a JSON dict
                return header, json.loads(data_decoded)
            except json.decoder.JSONDecodeError:  # if not JSON, it's a string
                return header, data_decoded
        except UnicodeDecodeError:  # if not a string, it is binary.
            return header, data

    def send(self, header, data, encoding="utf-8"):
        """Send data to the connected node.

        Args:
            header: JSON header of the packet.
            data: Actual meat of data.
            encoding: Encoding type.
        """
        msg = b""
        # encode header
        assert isinstance(header, dict)
        try:
            json_header = json.dumps(header)
            json_header = json_header.encode(encoding)
            # first 4 bytes are always for the header size
            msg += len(json_header).to_bytes(4, byteorder="big")
            msg += json_header
        except TypeError as type_error:
            print("Invalid header type (" + type_error + ").")
        except Exception as e:
            print("Unexpected error in header (" + e + ").")

        # send data
        if isinstance(data, str):
            msg += data.encode(encoding)

        elif isinstance(data, dict):
            try:
                json_data = json.dumps(data)
                msg += json_data.encode(encoding)

            except TypeError as type_error:
                print("Invalid dictionary type.")
                print(type_error)

            except Exception as e:
                print("Unexpected Error in send message")
                print(e)

        elif isinstance(data, bytes):
            msg += data

        else:
            raise ValueError("Unexpected data type.")

        # print("Sent:", len(msg))
        self.socket.sendall(len(msg).to_bytes(4, byteorder="big") + msg)

    # This method should be implemented by yourself!
    # We do not know when the message is
    # correct.
    # def check_message(self, data):
    #         return True

    def stop(self):
        self.kill_flag.set()

    def set_info(self, key, value):
        self.info[key] = value

    def get_info(self, key):
        return self.info[key]

    def __repr__(self):
        retVal = "Connection: {}:{}  <---->  {}:{}".format(
            self.node.host, self.node.port, self.host, self.port
        )
        return retVal

    def __str__(self):
        return self.__repr__()


def is_open_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def getfreeport():
    port = randint(49152, 65535)
    portsinuse = []
    while True:
        conns = psutil.net_connections()
        for conn in conns:
            portsinuse.append(conn.laddr[1])
        if port in portsinuse:
            port = randint(49152, 65535)
        else:
            break
    return port
