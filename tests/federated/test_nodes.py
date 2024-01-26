import imagiq.federated as iqf
from monai.data import CacheDataset
import time
from imagiq.models import Model
import torch.nn as nn
import torch.nn.functional as F
import pytest_check as check
import pytest

# from monai.networks.nets import densenet121, se_resnet50
# TODO: fixtures


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 60)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# TODO: Resume from the node previously created
def test_node_connection():
    """Tests if two nodes can connect with each other."""
    node1_port = 8000
    node2_port = 8001

    node1 = iqf.nodes.Node("localhost", node1_port)
    node2 = iqf.nodes.Node("localhost", node2_port)

    print(node1)
    print(node2)

    node1.start()
    node2.start()
    time.sleep(2)

    check.is_true(node1.is_alive())
    check.is_true(node2.is_alive())

    node1.connect_to("localhost", node2_port)
    time.sleep(1)

    check.equal(len(node1.peers_outbound), len(node2.peers_inbound))
    check.equal(len(node2.peers_outbound), len(node1.peers_inbound))

    node2.connect_to("localhost", node1_port)
    time.sleep(1)

    check.equal(len(node1.peers_outbound), len(node2.peers_inbound))
    check.equal(len(node2.peers_outbound), len(node1.peers_inbound))

    node1.stop()
    node2.stop()
    node1.join()
    node2.join()

    time.sleep(10)

    check.is_false(node1.is_alive())
    check.is_false(node2.is_alive())

    node1.destroy()
    node2.destroy()


def test_node_add_model():
    # Create a node
    node = iqf.nodes.Node("localhost", 8000)

    # Test if a model can be added.
    node.add_model(
        Model(Net()),
    )
    check.equal(len(node.model_bench), 1)

    # Test if a list of multiple models can be added.
    node.add_model(
        [
            Model(Net()),
            Model(Net()),
        ]
    )
    check.equal(len(node.model_bench), 3)

    # Clean up the node
    node.destroy()


def test_node_add_dataset():
    # Create a node
    node = iqf.nodes.Node("localhost", 8000)

    # Create a dummy dataset
    data = [
        {"image": [[0, 0, 0], [0, 0, 0]], "label": 0},
        {"image": [[1, 1, 1], [1, 1, 1]], "label": 1},
    ]
    dataset = CacheDataset(data, None)

    # Test if a dataset can be added.
    node.add_dataset(dataset)

    # Clean up the node
    node.destroy()


def test_send_model():
    node1_port = 8000
    node2_port = 8001
    node1 = iqf.nodes.Node("localhost", node1_port)
    node2 = iqf.nodes.Node("localhost", node2_port)
    node1.start()
    node2.start()
    time.sleep(2)

    node1.connect_to("localhost", node2_port)
    time.sleep(1)

    model1 = Model(Net())
    model2 = Model(Net())
    model3 = Model(Net())
    node1.add_model(
        [
            model1,
            model2,
            model3,
        ]
    )
    node1.commit_models("initial commit")

    # send a model by index
    node1.send_model(peer_index=0, model_index=0)
    time.sleep(2)
    assert len(node2.model_bench) == 1
    assert node2.model_bench[0].uid == model1.uid

    # wrong index raises an exception
    with pytest.raises(IndexError):
        node1.send_model(peer_index=0, model_index=99)

    # non existing peer raises an exception
    with pytest.raises(IndexError):
        node1.send_model(peer_index=1, model_index=0)

    # node2 is not connected to node 1, so won't be able to send a model.
    with pytest.raises(IndexError):
        node2.send_model(0, model_index=0)

    # send a model by uid
    node1.send_model(0, uid=model2.uid)
    time.sleep(2)
    assert len(node2.model_bench) == 2
    assert node2.model_bench[1].uid == model2.uid

    # non-existing uid raises an exception
    with pytest.raises(ValueError):
        node1.send_model(0, uid="invalid_uid")

    # send a model by name
    node1.send_model(0, name=model3.name)
    time.sleep(2)
    assert len(node2.model_bench) == 3
    assert node2.model_bench[2].uid == model3.uid

    # non-existing name raises an exception
    with pytest.raises(ValueError):
        node1.send_model(0, name="invalid_name")

    # if nothing is specified, complain
    with pytest.raises(ValueError):
        node1.send_model(peer_index=0)

    # if too many things are specified, complain
    with pytest.raises(ValueError):
        node1.send_model(0, model_index=0, uid=model1.uid)
    with pytest.raises(ValueError):
        node1.send_model(0, model_index=0, name=model1.name)
    with pytest.raises(ValueError):
        node1.send_model(0, uid=model1.uid, name=model1.name)
    with pytest.raises(ValueError):
        node1.send_model(0, model_index=0, uid=model1.uid, name=model1.name)

    node1.stop()
    node2.stop()
    node1.join()
    node2.join()
    time.sleep(15)

    node1.destroy()
    node2.destroy()


def test_model_broadcast():
    """Tests if two nodes can exchange files."""
    node1_port = 8000
    node2_port = 8001
    node1 = iqf.nodes.Node("localhost", node1_port)
    node2 = iqf.nodes.Node("localhost", node2_port)
    node1.start()
    node2.start()
    time.sleep(2)

    node1.connect_to("localhost", node2_port)
    time.sleep(1)
    node2.connect_to("localhost", node1_port)
    time.sleep(1)

    # Test if a list of multiple models can be added.
    node1.add_model(
        [
            Model(Net()),
            Model(Net()),
        ]
    )
    node1.commit_models("Initial commit")
    time.sleep(1)

    node1.broadcast_models()
    time.sleep(3)
    check.equal(len(node2.model_bench), len(node1.model_bench))

    # broadcast them again without commits
    for model in node1.model_bench:
        model.net = None
    node1.broadcast_models()
    time.sleep(3)
    check.equal(len(node2.model_bench), len(node1.model_bench))

    # broadcast them again after commits
    node1.commit_models("Second commit")
    node1.broadcast_models()
    time.sleep(3)
    for i in range(len(node1.model_bench)):
        check.equal(node1.model_bench[i], node2.model_bench[i])

    node1.stop()
    node2.stop()
    node1.join()
    node2.join()
    time.sleep(15)

    node1.destroy()
    node2.destroy()


# def test_create_federation(createNode):
#     node = createNode
#     name = "Test Federation"
#     config['description'] = "This is a test federation."
#     fed = node.create_federation(
#         name = name,
#         config = config
#         )
#     assert fed != None
#     # assert fed.getUID() is in node.federations
#     node.destroy()
#
#
# def test_join_federation():
#     node1 = iqf.nodes.Node()
#     node2 = iqf.nodes.Node()
#     node1.create()
#     node2.create()

#     fed = node1.create_federation(
#         name = "test",
#         config = None
#     )
#     approved = node2.join(fed.getUID())

#     node1.destroy()
#     node2.destroy()

# def test_leave_federation():
#     node1 = iqf.nodes.Node()
#     node2 = iqf.nodes.Node()
#     node1.create()
#     node2.create()

#     fed = node1.create_federation(
#         name = "test",
#         config = None
#     )
#     approved = node2.join(fed.getUID())
#     node2.leave(fed.getUID())

#     node1.destroy()
#     node2.destroy()


# def test_send_want_list():
#     node1 = iqf.nodes.Node()
#     node2 = iqf.nodes.Node()
#     node1.create()
#     node2.create()

#     fed = node1.create_federation(
#         name = "test",
#         config = None
#     )
#     approved = node2.join(fed.getUID())
#     node2.send_want_list(fed.getUID())
