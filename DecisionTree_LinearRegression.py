
import numpy as np
from sklearn import linear_model as lm
from graphviz import Digraph, nohtml

class Node():
    
    """
    A class representing a single node of the entire tree
    
    ...
    
    Attributes
    ----------
    
    feature_idx : int
        The index of the feature where the node is being split
        
    pivot_value : float/int
        The value of the feature_idx where the node is split
        
    model : sklearn model
        The linear regression model for the node
        
    depth : int
        The current depth of the node (Root node is 0)
        
    left_node : Node object
        If the current node is not a leaf node, it stores the entire left child node
        
    right_node : Node object
        If the current node is not a leaf node, it stores the entire right child node
        
    model_intercept : float 
        Stores the model intercept for the leaf nodes (default None)
        
    model_coef : np array
        Stores the model coeffcients for the leaf nodes (default None)
    
    """
    
    def __init__(self,feature_idx,pivot_value,linear_model,depth):
        
        self.feature_idx = feature_idx
        self.pivot_value = pivot_value
        self.model = linear_model
        self.depth = depth
        self.left_node = None
        self.right_node = None  
        self.model_intercept = None
        self.model_coef = None
        self.parent_node = None
        self.unique_id = 0
        
class LinearModelTree():
    
    """
    A class representing the entire decision tree 
    
    ...
    
    Attributes
    ----------
    
    reg_features : list
        A list of feature indices (a subset of X) that is used to regress onto the output variable
        
    max_depth : int
        The maximum depth of the tree (default None)
        
    min_samples_split : int
        Minimum number of samples required in a node to perform a split (default 100)
        
    min_sampls_leaf : int
        Minimum number of samples that constitute a leaf (default 50)
        
    model_type : str
        Model used for linear regression (default 'Ridge')
        
    num_cat : int
        Maximum number of unique variables to treat a feature as a categorical variable (default 2)
        
    num_cont : int
        Maximum number of pivot points to check while performing a split for a continuous feature (default 100)
        
    current_depth : int
        The current depth of the node under consideration
        
    depth : list
        A list of every split made denoted by the current depth of the node
    
    
    Methods
    -------
    
    build_tree(X,y,depth=0)
        Builds the entire tree from the root node to leaf nodes from the given data
        
    best_split(X,y)
        Given the data, it finds the best possible split based on the linear split in the child nodes
        
    split_node(X,y,feature_idx,pivot_value)
        Given the data, the index of the feature, and the pivot-value it returns the split child nodes
        
    feature_type(X,num_cat)
        Given X and the num_cat, it returns all the features that are type categorical
        
    pivot_value_dictionary(X)
        Given the input array X, it returns the pivot values for all the features
        
    fit(X,y)
        Returns the final tree fit with the training data X,y
        
    predict_one(X,cat)
        Given a single data point and the set of categorical features it returns the prediction
        
    predict(X)
        Returns the prediction for a collection of data points (could use for testing set)
        
    RMSE(X,y)
        Returns the root mean squared error for a given X,y
        
    tree_param(mytree,tree_val=[])
        Given the final tree, it returns a list of regression coefficients at each node
        
    get_depth()
        Returns the maximum depth achieved by the final tree
    
    
    """
    
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
        self.current_id = 0
        self.depth = [0]
        self.leaves_count = 0
        
    def raise_errors(self):
        
        if not type(self.reg_features) is list:
            raise TypeError('reg_features should be of type list!')
        if not type(self.num_cont) is int:
            raise TypeError('num_cont should be of type int!')
        
    
    def build_tree(self,X,y,depth=0,current_node=None):
        
        """
        Returns the tree built from the root to leaves
        
        Parameters
        ----------
        
        X : numpy array
            Predictors or input
        y : numpy array shape (-1,1)
            Output
        depth : int
            Used to mark the current depth of the node
        """
        
        self.current_depth = depth
        
        if self.max_depth == None:
            cond = X.shape[0] >= self.min_samples_split
        else:
            cond = X.shape[0] >= self.min_samples_split and self.current_depth < self.max_depth
            
        if cond:
            split_dict = self.best_split(X,y)
            
            if split_dict['feature'] != 'Leaf':
                node = Node(split_dict['feature'],split_dict['pivot_value'],split_dict['model'],depth)
                node.parent_node = current_node
                self.current_id = self.current_id + 1
                node.unique_id = self.current_id
                self.current_depth = node.depth + 1
                self.depth.append(self.current_depth)
                (X_l,y_l),(X_r,y_r) = split_dict['split_data']
                node.left_node = self.build_tree(X_l,y_l,depth+1,node)
                node.right_node = self.build_tree(X_r,y_r,depth+1,node)
                
            else:
                self.leaves_count += 1
                reg_model = lm.Ridge().fit(X[:,self.reg_features],y.reshape(-1,1))
                node = Node('Leaf','None',reg_model,depth)
                node.parent_node = current_node
                self.current_id = self.current_id + 1
                node.unique_id = self.current_id
                node.model_intercept = reg_model.intercept_
                node.model_coef = reg_model.coef_
                self.current_depth = node.depth

        else:
            self.leaves_count += 1
            reg_model = lm.Ridge().fit(X[:,self.reg_features],y.reshape(-1,1))
            node = Node('Leaf','None',reg_model,depth)
            node.parent_node = current_node
            self.current_id = self.current_id + 1
            node.unique_id = self.current_id
            node.model_intercept = reg_model.intercept_
            node.model_coef = reg_model.coef_
            self.current_depth = node.depth

        return node
    
    
    def best_split(self,X,y):
        
        """
        Returns a dictionary of feature index, pivot value, and model
        
        Parameters
        ----------
        
        X : numpy array
            Predictors or input
        y : numpy array shape (-1,1)
            Output
        """
    
        features_total = X.shape[1]
        r2 = []
        feat_pivot = []
        model_l_store = []
        model_r_store = []       
        
        for feature_idx in range(0,features_total):
            
            pivot_values = self.pivot_dict[feature_idx]
            
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
        
        """
        Returns the child left nodes and right nodes
        
        Parameters
        ----------
        
        X : numpy array
            Predictors or input
        y : numpy array shape (-1,1)
            Output
        feature_idx : int
            The index of the feature that is used to create the split
        pivot_value : float/int
            The value of the feature at which the split is made
        """
        
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
        
        """
        Returns a list of feature indices that are of type categorical
        
        Parameters
        ----------
        
        X : numpy array
            Predictors or input
        num_cat : int
            Maximum number of unique values in a feature to treat it as a categorical type 
        """
        
        cat = []
        features_total = X.shape[1]
        
        for feature_idx in range(0,features_total):
            if np.unique(X[:,feature_idx]).shape[0] <= num_cat:
                cat.append(feature_idx)
                
        return cat  
    
    
    def pivot_value_dictionary(self,X):
        
        """
        Returns a dictionary of feature indices as keys and pivot values as values of the dictionary
        
        Parameters
        ----------
        X : numpy array
            Input or predictors of the training data

        """
        
        cat = self.feature_type(X,self.num_cat)
        features_total = X.shape[1]
        pivot_dict = {}
        
        for feature_idx in range(0,features_total):
            
            if feature_idx in cat:
                pivot_dict[feature_idx] = np.unique(X[:,feature_idx])
            else:
                pivot_dict[feature_idx] = np.round(np.linspace(X[:,feature_idx].min(),X[:,feature_idx].max(),self.num_cont),3)
        
        return pivot_dict
    
        
    def fit(self,X,y):
        
        """
        Returns the final tree fit with the training data
        
        Parameters
        ----------
        
        X : numpy array
            Input or predictors of the training data
        y : numpy array shape (-1,1)
            Output of the training data
            
        """
        
        self.raise_errors()
        
        self.pivot_dict = self.pivot_value_dictionary(X)
        
        self.final_tree = self.build_tree(X,y)
        
        return self
        
        
    def predict_one(self,X,cat):
        
        """
        Returns predictions for a single data point
        
        Paramters
        ---------
        
        X : 1-d numpy array
            A single input data point
        cat : list
            List of all categorical feature indices
        """
            
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
        
        """
        Returns prediction for a set of inputs
        
        Paramters
        ---------
        
        X : numpy array
            Predictors or input
        """
        
        cat = self.feature_type(X,self.num_cat)
        predictions = []
        
        for Xi in X:
            predictions.extend(self.predict_one(Xi,cat))
        
        return predictions
    
    
    def RMSE(self,X,y):
        
        """
        Returns the root mean squared error
        
        Parameters
        ----------
        
        X : numpy array
            Predictors or input
        y : numpy array shape (-1,1)
            Output
        """
        
        return np.sqrt((1/X.shape[0])*np.sum(np.square(self.predict(X)-y.reshape(-1,1))))
    
    
    def tree_param(self,mytree,columns,tree_val = None):
        
        """
        Returns the entire structure of the tree along with the split and model parameters
        
        (feature index, pivot value, linear model intercept, linear model slopes)
        
        Parameters
        ----------
        
        mytree : Node object 
                A node object containing the tree, subtree or leaf node
        columns : list of str
                A list of strings denoting the column names for the input matrix X
        tree_val : list
                A list containing the split and model parameters of the entire tree
        """
        
        if tree_val is None:
            tree_val = []
        
        if mytree:
            try:
                if mytree.model_intercept is None:
                    tree_val.append([(mytree.unique_id,columns[mytree.feature_idx],mytree.pivot_value),(columns[mytree.parent_node.feature_idx],mytree.parent_node.pivot_value)])
                else:
                    tree_val.append([(mytree.unique_id,columns[mytree.feature_idx],mytree.pivot_value,mytree.model_intercept,mytree.model_coef,columns[mytree.parent_node.feature_idx],mytree.parent_node.pivot_value)])
            except AttributeError:
                if mytree.model_intercept is None:
                    tree_val.append([(mytree.unique_id,columns[mytree.feature_idx],mytree.pivot_value)])
                else:
                    tree_val.append([(mytree.unique_id,columns[mytree.feature_idx],mytree.pivot_value,mytree.model_intercept,mytree.model_coef)])
            except TypeError:
                    tree_val.append([(mytree.unique_id,mytree.feature_idx,mytree.model_intercept,mytree.model_coef)])
            tree_val = self.tree_param(mytree.left_node,columns,tree_val)
            tree_val = self.tree_param(mytree.right_node,columns,tree_val)
        return tree_val
    
    
    def plot_tree(self,mytree,columns,g=None):
        
        if g is None:
            g = Digraph('g',node_attr={'fontsize':'12', 
                         'shape':'box', 
                         'color':'black'})
            
        if mytree:
            try:
                g.node(str(mytree.unique_id),f'{(columns[mytree.feature_idx],mytree.pivot_value)}',color='lightsteelblue3',style='filled')
                g.node(str(mytree.parent_node.unique_id),f'{(columns[mytree.parent_node.feature_idx],mytree.parent_node.pivot_value)}',color='lightsteelblue3',style='filled')
                g.edge(str(mytree.parent_node.unique_id),str(mytree.unique_id))
            except:
                if mytree.feature_idx == 'Leaf':
                    g.node(str(mytree.unique_id),'Leaf',color='darkolivegreen',style='filled')
                    g.node(str(mytree.parent_node.unique_id),f'{(columns[mytree.parent_node.feature_idx],mytree.parent_node.pivot_value)}',color='lightsteelblue3',style='filled')
                    g.edge(str(mytree.parent_node.unique_id),str(mytree.unique_id))
                else:
                    pass
            g = self.plot_tree(mytree.left_node,columns,g)
            g = self.plot_tree(mytree.right_node,columns,g)
        return g
                                                     
                       
    def get_n_leaves(self):
        
        """
        Returns the number of leaf nodes in the final tree
        """                    
        
        return self.leaves_count

                                                                     
    def get_depth(self):
        
        """
        Returns the maximum depth of the generated tree
        """
        
        return  max(self.depth)
    
    
                                         
                                                     
            