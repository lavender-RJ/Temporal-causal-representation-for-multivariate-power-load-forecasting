from matplotlib.pyplot import SubplotSpec
import numpy as np
from sklearn.metrics import r2_score
def calculate_imse(y_true_lower, y_true_upper, y_pred_lower, y_pred_upper):
    imse = np.mean(np.square(y_pred_upper- y_true_upper + y_pred_lower- y_true_lower))
    return imse

def calculate_imae(y_true_lower, y_true_upper, y_pred_lower, y_pred_upper):
    imae = np.mean(np.abs(y_pred_upper-y_true_upper) + np.abs(y_pred_lower-y_true_lower))
    return imae

def calculate_isse(y_true_lower, y_true_upper, y_pred_lower, y_pred_upper):
    sspe = np.sum(np.square(y_pred_upper- y_true_upper + y_pred_lower- y_true_lower))
    return sspe

def calculate_imspe(y_true_lower, y_true_upper, y_pred_lower, y_pred_upper):
    impe = np.mean(np.abs((np.maximum(y_pred_upper, y_true_upper) - np.minimum(y_pred_lower, y_true_lower)) / (y_true_upper - y_true_lower)))
    return impe * 100  # 转换为百分比
    
def metric3(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape
   
def CORR_np(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        #B, N
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        #np.transpose include permute, B, T, N
        pred = np.expand_dims(pred.transpose(0, 2, 1), axis=1)
        true = np.expand_dims(true.transpose(0, 2, 1), axis=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(0, 1, 2, 3)
        true = true.transpose(0, 1, 2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(axis=dims)
    true_mean = true.mean(axis=dims)
    pred_std = pred.std(axis=dims)
    true_std = true.std(axis=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(axis=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation

def calculate_r2(pred, true):
    r2 = r2_score(true.reshape(-1,1), pred.reshape(-1,1))
    return r2

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def huber_loss(y, y_hat, delta=1):
    residuals = np.abs(y - y_hat)
    quadratic = np.minimum(residuals, delta)
    linear = residuals - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return loss
 
def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = calculate_r2(pred, true)
    # huber = huber_loss(pred, true)
    
    return mae,mse,rmse,mape,mspe,r2
# loss = criterion(true.detach().cpu()[:,:,-1:].flatten(),true.detach().cpu()[:,:,-2:-1].flatten(),pred.detach().cpu()[:,:,-1:].flatten(),pred.detach().cpu()[:,:,-2:-1].flatten())
def metric1(pred, true):
    imae = calculate_imae(true[:,:,-1:].flatten(),true[:,:,-2:-1].flatten(),pred[:,:,-1:].flatten(),pred[:,:,-2:-1].flatten())
    imse = calculate_imse(true[:,:,-1:].flatten(),true[:,:,-2:-1].flatten(),pred[:,:,-1:].flatten(),pred[:,:,-2:-1].flatten())
    imspe =calculate_imspe(true[:,:,-1:].flatten(),true[:,:,-2:-1].flatten(),pred[:,:,-1:].flatten(),pred[:,:,-2:-1].flatten())
    isse = calculate_isse(true[:,:,-1:].flatten(),true[:,:,-2:-1].flatten(),pred[:,:,-1:].flatten(),pred[:,:,-2:-1].flatten())
    # huber = huber_loss(pred, true)
    
    return imae,imse,imspe,isse