from models import HAR_Mamba
from  models import iTransformer
from models import PatchTST

def get_model(arg):

    if arg.dataset_name ==  'har70+':    
        num_features = 6      
        num_classes = 7 
        
    if arg.model_name == 'mamba':   
        hidden_dim = arg.hidden_dim    
        # Instantiate model
        model = HAR_Mamba(input_dim=num_features, hidden_dim=hidden_dim, num_classes=num_classes)
        return model
    
    if arg.model_name == 'itransformer':
        model = iTransformer.Model(arg)
        return model
    
    if arg.model_name == 'patchtst':
        model = PatchTST.Model(arg)
        return model

    
