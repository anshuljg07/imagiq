from .common import NODES_DIR
import os

from .core import Node
from imagiq.federated.connections import is_open_port, getfreeport


def dir():
    """List all local nodes.

    Return:
        list: A list of node objects.
    """
    if os.path.exists(str(NODES_DIR)):
        node_list = []
        for uid in os.listdir(str(NODES_DIR)):
            port = getfreeport()
            node = Node("localhost", port, uid=uid)
            node.load(uid)
            node_list.append(node)
        return node_list
    else:
        return []
