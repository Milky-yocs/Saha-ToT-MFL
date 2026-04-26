import numpy as np
import logging
import time

class ClientAssociation(object):
    """Client association decision making."""
    def __init__(self, asso_type, model_name,
                 pref=None, cls_num=None, labels=None):
                                                     
                                                          
        self.asso_type = asso_type
        self.pref = pref
        self.labels = labels
        self.cls_num = cls_num
        self.label_num = len(labels) if labels is not None else None
        self.asso_time = 0.0

    def solve(self, conn_ub, grads=None, num_samples=None, R=None,
              R_ub=None, phi=None):
        """
        Solve client association.
        Paper-delivery subset only keeps random association.
        Suppose there are N devices and G gateways.

        Args:
            conn_ub: [N, G] matrix, feasible connections
            grads: [N, # of weights], latest gradients at N clients
            num_samples: [N], number of samples at N clients
            R: [N, G] matrix, throughput of all possible links between
                device and gateway
            R_ub: [G], upper bound of throughput at each gateway
            phi: reserved for compatibility, unused in random association

        Returns:
            conn: [N, G] matrix, decided connection
        """
        N = conn_ub.shape[0]
        G = conn_ub.shape[1]

        start = time.time()

                                  
        num_samples = num_samples.reshape((-1, 1))          
        global_grad = np.sum(
            np.multiply(grads, num_samples), axis=0
        ) / np.sum(num_samples)        

                                    
        dissimil_mat = grads @ grads.T          
        np.fill_diagonal(dissimil_mat, 0.0)

        eta = (global_grad @ grads.T)         
        v = - np.sum(dissimil_mat, axis=1) / (N - 1)        
        u = eta + v

                               
        R_ratio = np.divide(R, R_ub.reshape((1, -1)))          

        if self.asso_type != 'random':
            raise ValueError(
                "client association type not supported in paper subset: {}".format(
                    self.asso_type
                )
            )

        conn = []
        for i in range(N):
            avail_ids = np.where(conn_ub[i])[0]
                                     
            gateway_id = np.random.choice(avail_ids)
            conn.append(np.eye(G)[gateway_id])

        conn = np.array(conn, dtype=int)

        logging.info('Obj 1: {}'.format(u @ conn))
        logging.info('Obj 2: {}'.format(np.diag(R_ratio.T @ conn)))

        cur_asso_time = time.time() - start
        self.asso_time += cur_asso_time
        logging.info('association time: {} accu time: {}'.format(cur_asso_time, self.asso_time))

        return conn
