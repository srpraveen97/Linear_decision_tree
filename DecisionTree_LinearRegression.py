
import numpy as np
import pandas as pd
from sklearn import linear_model as lm

class Node():
    
    def __init__(self,feature_idx,pivot_value,linear_model,depth):
        
        self.feature_idx = feature_idx
        self.pivot_value = pivot_value
        self.model = linear_model
        self.depth = depth
        self.left_node = None
        self.right_node = None  
        
class LinearModelTree():
    
    def __init__(self,reg_features,max_depth=None,min_samples_split=100,min_samples_leaf=50,
                 model_type='Ridge',num_cat=2,num_cont=100):
        
        self.reg_features = reg_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model_type = model_type
        self.num_cat = num_cat
        self.num_cont = num_cont
        self.current_depth = 0
        self.depth = [0]
        self.final_depth = 0
    
    def build_tree(self,X,y,depth=0):
        
        self.current_depth = depth
        
        if self.max_depth == None:
            cond = X.shape[0] >= self.min_samples_split
        else:
            cond = X.shape[0] >= self.min_samples_split and self.current_depth < self.max_depth
        
        if cond:
            split_dict = self.best_split(X,y)
            
            if split_dict['feature'] != 'Leaf':
                node = Node(split_dict['feature'],split_dict['pivot_value'],split_dict['model'],depth)
                self.current_depth = node.depth + 1
                self.depth.append(self.current_depth)
                (X_l,y_l),(X_r,y_r) = split_dict['split_data']
                node.left_node = self.build_tree(X_l,y_l,depth+1)
                node.right_node = self.build_tree(X_r,y_r,depth+1)
                
            else:
                reg_model = lm.Ridge().fit(X[:,self.reg_features].reshape(-1,len(self.reg_features)),y.reshape(-1,1))
                node = Node('Leaf','None',reg_model,depth)       
                self.current_depth = node.depth
        
        else:
            reg_model = lm.Ridge().fit(X[:,self.reg_features].reshape(-1,len(self.reg_features)),y.reshape(-1,1))
            node = Node('Leaf','None',reg_model,depth)
            self.current_depth = node.depth
            
        self.final_depth = max(self.depth)
            
        return node
    
    
    def best_split(self,X,y):
    
        features_total = X.shape[1]
        r2 = []
        feat_pivot = []
        model_l_store = []
        model_r_store = []
        
        for feature_idx in range(0,features_total):
            
            cat = self.feature_type(X,self.num_cat)
            
            if feature_idx in cat:
                pivot_values = np.unique(X[:,feature_idx])
            else:
                pivot_values = np.linspace(X[:,feature_idx].min(),X[:,feature_idx].max(),self.num_cont)
            
            for pivot_value in pivot_values:
                
                (X_l,y_l),(X_r,y_r) = self.split_node(X,y,feature_idx,pivot_value)
                
                if X_l.shape[0] >= self.min_samples_leaf and X_r.shape[0] >= self.min_samples_leaf:
                    
                    linear_model = lm.Ridge()
                    
                    model_l = linear_model.fit(X_l[:,self.reg_features],y_l.reshape(-1,1))
                    
                    linear_model = lm.Ridge()
                    
                    model_r = linear_model.fit(X_r[:,self.reg_features],y_r.reshape(-1,1))
                    
                    model_l_store.append(model_l)
                    model_r_store.append(model_r)

                    avg_r2 = 0.5*np.abs(model_l.score(X_l[:,self.reg_features],y_l)) + 0.5*np.abs(model_r.score(X_r[:,self.reg_features],y_r))
                    r2.append(avg_r2)
                        
                    feat_pivot.append((feature_idx,pivot_value))
        
        try:
            feature_idx,pivot_value = feat_pivot[np.argmax(r2)]
            return {'feature':feature_idx,'pivot_value':pivot_value,'split_data':self.split_node(X,y,feature_idx,pivot_value),'model':(model_l_store[np.argmax(r2)],model_r_store[np.argmax(r2)])}
        except:
            return {'feature':'Leaf','pivot_value':'None','split_data':'NoSplit','model':lm.Ridge().fit(X[:,self.reg_features].reshape(-1,len(self.reg_features)),y.reshape(-1,1))}

            
    def split_node(self,X,y,feature_idx,pivot_value):
        
        cat = self.feature_type(X,self.num_cat)
    
        if feature_idx in cat:

            left_node = (X[X[:,feature_idx]!=pivot_value],y[X[:,feature_idx]!=pivot_value])
            right_node = (X[X[:,feature_idx]==pivot_value],y[X[:,feature_idx]==pivot_value])

        else:
            
            index = X[:,feature_idx] >= pivot_value
            left_node = (X[~index],y[~index])
            right_node = (X[index],y[index])

        return left_node,right_node
    
    @staticmethod
    def feature_type(X,num_cat):
        
        cat = []
        features_total = X.shape[1]
        
        for feature_idx in range(0,features_total):
            if np.unique(X[:,feature_idx]).shape[0] <= num_cat:
                cat.append(feature_idx)
                
        return cat  
        
        
    def fit(self,X,y):
        
        self.final_tree = self.build_tree(X,y)
        
        return self
        
        
    def predict_one(self,X,cat):
            
        mytree = self.final_tree

        while mytree.left_node:
            if mytree.feature_idx in cat:
                if X[mytree.feature_idx] == mytree.pivot_value:
                    mytree = mytree.right_node
                else:
                    mytree = mytree.left_node
            else:
                if X[mytree.feature_idx] >= mytree.pivot_value:
                    mytree = mytree.right_node
                else:
                    mytree = mytree.left_node
                    
        return mytree.model.predict(X[self.reg_features].reshape(-1,len(self.reg_features)))
        
    
    def predict(self,X):    
        
        cat = self.feature_type(X,self.num_cat)
        predictions = []
        
        for Xi in X:
            predictions.extend(self.predict_one(Xi,cat))
        
        return predictions
    
    
    def RMSE(self,X,y):
        
        return np.sqrt((1/X.shape[0])*np.sum(np.square(self.predict(X)-y.reshape(-1,1))))
                                                     
                                                                                            
    def get_depth(self):
        
        return self.final_depth
    
    
    def get_leaves(self):
        
        pass
    
    
    def get_params(self):
        
        pass
    
    
    def entire_tree(self):
        
        pass
                                         
                                                     
            