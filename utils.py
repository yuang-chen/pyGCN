import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    """ ^^^
    The numpy.identity (identity array) is a square array with ones on the main diagonal
    """
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    """ ^^^
    id:         idx_features_labels[:, 0]
    features:   idx_features_labels[:, 1:-1]
    labels:     idx_features_labels[:, -1]
    """
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    """ ^^^
    compress the sparse matrix into the csr representation to save memory space
    """
    labels = encode_onehot(idx_features_labels[:, -1])
    """ ^^^
    using one-hot vector to represent the label`s categorties
    """
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    """ ^^^
    using a hashmap to sort and store the id - {id: count}
    e.g., id(1061127): count(1234)
    """
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    """ ^^^
    get an array[num_edges][2], 
    edges_unordered[*][0]: source_id
    edges_unordered[*][1]: destination_id  
    """
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    """ ^^^
    flatten the edges_unordered: [num_edges][2] -> [num_edges*2]
    apply the idx_map.get() function on the flattened array 
    reshape the modified array: [num_edges*2] -> [num_edges][2]

    """
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    """ ^^^
    the adjacent matrix get here is directional
    """
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    """ ^^^
    convert the directional matrix to undirectional (symmetric) matrix
    the multiply() here is point(element)-wise multiplication
    """
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    """ ^^
    row normalize
    """

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    """ ^^^
    get back the original 2-dimensional sparse matrix and convert it to a tensor
    """
    labels = torch.LongTensor(np.where(labels)[1])
    """ ^^^
    convert the 2-dimensional one-hot vector to one-dimensional tensor,
    where the value of each number representing the class
    """
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
    """ ^^^
    final form of returned variables (all are tensors):
    adj:       a symmetric sparse matrix with normalized row   [num_vertex, num_vertex] 
    features:  a sparse matrix with normalized row             [num_vertex, num_feature]
    labels:    a sparse matrix                                 [num_vertex, 1]
    """

def normalize(mx):
    """Row-normalize sparse matrix"""
    """
    sum up the matrix along the Row axis
    inverse the sum and flatten it into an one-dimension array
    construct a diagonal matrix based on the r_inv
    in the final matrix mx, the sum of each row is 1
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

    

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tmp = torch.sparse_coo_tensor(indices, values, shape)
    return torch.sparse.FloatTensor(indices, values, shape)
