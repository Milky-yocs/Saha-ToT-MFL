import logging
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import time
import copy
from sklearn.decomposition import PCA

class Client(object):
    """Simulated federated learning client."""

    def __init__(self, client_id):
        self.client_id = client_id
        self.available = True                                              
                                                    
                                                                          
        self.loss = None
        self.delay = None
        self.pca = None
                                                                      
        self.participation_count = 0
        self.last_participation_round = -1
        self.cache_version = -1
        self.cache_hash = ""
        self.cache_source_aggregator_id = -1
        self.cache_ready = False
        self.cache_ready_time = 0.0

    def __repr__(self):
                                                              
                                                                                    
        return 'Client #{}'.format(self.client_id)

                                     
    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias

    def set_shard(self, shard):
        self.shard = shard

    def set_cls_num(self, cls_num):
        self.cls_num = cls_num

    def _create_optimizer(self, fl_model):
        self.optimizer = fl_model.get_optimizer(self.model)

                         
    def download(self, argv):
                                   
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv):
                              
        try:
            return argv.copy()
        except:
            return argv

    def set_available(self):
        self.available = True

    def set_unavailable(self):
        self.available = False

                               
    def set_data(self, data, config):
                             
        do_test = self.do_test = config.clients.do_test
        test_partition = self.test_partition = config.clients.test_partition
        self.dataset = config.model

                       
        self.data = self.download(data)

                                                   
        data = self.data
        if do_test:                                       
            self.trainset = data[:int(len(data) * (1 - test_partition))]
            self.testset = data[int(len(data) * (1 - test_partition)):]
        else:
            self.trainset = data

                         
        self.num_samples = len(data)

                               
    def set_data_leaf(self, train_data, test_data, config):
                             
        do_test = self.do_test = config.clients.do_test
        test_partition = self.test_partition = config.clients.test_partition
        self.dataset = config.model

                       
        self.data = self.download(train_data)

                                                   
        if do_test:                                       
            self.trainset = train_data
            self.testset = test_data
        else:
            self.trainset = train_data

                                                                        
        self.total_modalities = float(max(1.0, train_data.get('total_modalities', 3.0)))
        self.available_modalities = float(np.clip(
            train_data.get('available_modalities', self.total_modalities),
            0.0, self.total_modalities
        ))
        self.missing_modalities = float(np.clip(
            train_data.get('missing_modalities', self.total_modalities - self.available_modalities),
            0.0, self.total_modalities
        ))
        self.missing_modal_ratio = float(np.clip(
            train_data.get('missing_modal_ratio', self.missing_modalities / max(1.0, self.total_modalities)),
            0.0, 1.0
        ))
        self.s_modal = float(np.clip(
            train_data.get('s_modal', self.available_modalities / max(1.0, self.total_modalities)),
            0.0, 1.0
        ))

                         
        self.num_samples = len(train_data['x'])

    def set_gateway(self, gateway_id):
        self.gateway_id = gateway_id

    def set_delay_uniform(self):
                                                           
                                                        
        link_speed = max(random.normalvariate(self.speed_mean, self.speed_std), 10.0)
        comp_time = max(random.normalvariate(self.comp_mean, self.comp_std), 1.0)
        self.delay = (self.model_size / link_speed) + comp_time                       
        logging.debug('client {} link speed: {} comp time: {} delay: {}'.format(
            self.client_id, link_speed, comp_time, self.delay
        ))
                                                        

    def set_delay_to_gateway(self, comm_delay, comp_delay, model_size,
                             speed_mean=None, speed_std=None, comp_std=None):
                        
        self.model_size = model_size

                                                  
        self.speed_mean = speed_mean
        self.speed_std = speed_std

                            
        self.comp_mean = comp_delay
        self.comp_std = comp_std

                             
        self.delay = self.est_delay = comm_delay + comp_delay
        self.throughput = self.model_size / self.delay

    def sync_global_configure(self, config):
        import fl_model                                

                             
        self.model_path = config.paths.model

                              
        config = self.download(config)

                                                   
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

                                           
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global'
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Client {} load global model: {}'.format(
            self.client_id, path))

                          
        self._create_optimizer(fl_model)

    def async_global_configure(self, config, download_time):
        import fl_model                                

                             
        self.model_path = config.paths.model

                              
        config = self.download(config)

                                                   
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

                                           
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global_{}'.format(download_time)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Client {} load global model: {}'.format(
            self.client_id, path))

                          
        self._create_optimizer(fl_model)


    def sync_client_configure(self, config):
        import fl_model                                

                             
        self.model_path = config.paths.model

                              
        config = self.download(config)

                                                   
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

                                            
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/gateway{}'.format(self.gateway_id)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.debug('Client {} load gateway {} model: {}'.format(
            self.client_id, self.gateway_id, path))

                          
        self._create_optimizer(fl_model)

                                                                     
        if config.delay_mode == 'uniform':
            self.set_delay_uniform()

    def sync_client_configure_from_state(self, config, model_state, gateway_id=None):
        import fl_model                                

                             
        self.model_path = config.paths.model

                              
        config = self.download(config)

                                                   
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

                                                                     
        self.model = fl_model.Net()
        self.model.load_state_dict(copy.deepcopy(model_state))
        self.model.eval()
        if gateway_id is None:
            gateway_id = self.gateway_id
        logging.debug('Client {} load gateway {} model from in-memory state'.format(
            self.client_id, gateway_id))

                          
        self._create_optimizer(fl_model)

                                                                     
        if config.delay_mode == 'uniform':
            self.set_delay_uniform()


    def async_client_configure(self, config, gateway_download_time):
        import fl_model                                

                             
        self.model_path = config.paths.model

                              
        config = self.download(config)

                                                   
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

                                            
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/gateway{}_{}'.format(self.gateway_id,
                                                         gateway_download_time)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.debug('Client {} load gateway {} model: {}'.format(
            self.client_id, self.gateway_id, path))

                          
        self._create_optimizer(fl_model)

                                                                     
        if config.delay_mode == 'uniform':
            self.set_delay_uniform()

    def run(self, reg=None, rho=None):
                                         
        {
            "train": self.train(reg=reg, rho=rho)
        }[self.task]

    def get_report(self):
                                   
        return self.upload(self.report)

                            
    def train(self, reg=None, rho=None):
        import fl_model                                

        old_weights = fl_model.extract_weights(self.model)

                                
        trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
        self.loss = fl_model.train(
            self.model,
            trainloader,
            self.optimizer,
            self.epochs,
            reg,
            rho
        )

                                                                              
                                                    

                                          
        self.weights = fl_model.extract_weights(self.model)
        if self.dataset == 'Shakespeare' or self.dataset == 'HPWREN':
                                                   
            self.grads = self.extract_delta_weights(self.weights, old_weights)
        else:
            self.grads = fl_model.extract_grads(self.model)

                                                             
        if self.pca is not None:
            self.grads = self.flatten_weights(self.grads).reshape((1, -1))
            self.grads = self.pca.transform(self.grads).reshape(-1)
        else:
            self.grads = self.flatten_weights(self.grads)

                                     
        self.report = Report(self, self.weights, self.grads, self.loss, self.delay)


    def test(self, model):
                                                        
        import fl_model

        testloader = fl_model.get_testloader(self.testset, self.batch_size)
        test_loss, accuracy = fl_model.test(model, testloader)

        self.report.test_loss = test_loss
        self.report.accuracy = accuracy


    def extract_delta_weights(self, new_weights, old_weights):
                                                               
        deltas = []
        for i, (name, w) in enumerate(new_weights):
            bl_name, baseline = old_weights[i]

                                                    
            assert name == bl_name

                              
            delta = w - baseline
            deltas.append((name, delta))

        return deltas

                       
    @staticmethod
    def flatten_weights(weights):
                                      
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)


class Report(object):
    """Federated learning client report."""
    def __init__(self, client, weights, grads, loss, delay):
        self.client_id = client.client_id
        self.num_samples = client.num_samples
        self.weights = weights
        self.grads = grads
        self.loss = loss
        self.delay = delay

