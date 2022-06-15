from utils import get_FAIR
from models import FairFlow, ConditionalFairFlow
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Constants
SOURCE = "glaciers"
DATASET = "all"
REGION = 5  # Greenland
TRAINING_EPOCHS = 10
BATCH_SIZE = 250
NUM_FLOW_TRANSFORMS = 6
FLOW_HIDDEN_FEATURES = 32

# Data
start = time.time()
X, y, orig_df = get_FAIR(source=SOURCE, dataset=DATASET, region=REGION)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)

# Initialize Conditional Fair Flow and train
ff = ConditionalFairFlow(X=X_train, y=y_train, num_flow_transforms=NUM_FLOW_TRANSFORMS,
                         flow_hidden_features=FLOW_HIDDEN_FEATURES)
ff.train(num_epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, verbose=True, plot_logprob=False)

# Sample new values from learned distribution
samples, log_probs = ff.sample(num_samples=1000)

# See results
ff.visualize(plot='hist', samples=samples)

# Predict on new values
fxn = ff.calculate_mean_function(samples=samples, fit_type='nearest_neighbor')
predictions = ff.predict(context=X_test, fxn=fxn)

mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
mse = mean_squared_error(y_true=y_test, y_pred=predictions, squared=True)
rmse = mean_squared_error(y_true=y_test, y_pred=predictions, squared=False)
print(f"""MAE: {round(mae, 6)}
MSE: {round(mse, 6)}
RMSE: {round(rmse, 6)}""")

plt.hist((predictions - y_test), bins=20)
plt.title('Mean Function Residuals')
plt.xlabel('Absolute Error (Predicted - True)')
plt.ylabel('Frequency')
plt.xlim([-3,3])
plt.show()

plt.scatter(y_test, predictions, s=3)
plt.plot(y_test, y_test, 'r-')
plt.xlabel('Simulated SLE (cm)')
plt.ylabel('Emulated SLE (cm)')
plt.title('Mean Function Emulator')
plt.show()

plt.hist2d(y_test, predictions, bins=(20, 10))
plt.plot(y_test, y_test, 'r-',)
plt.xlabel('Simulated SLE (cm)')
plt.ylabel('Emulated SLE (cm)')
plt.title('Mean Function Emulator')
plt.show()

finish = time.time()
total_time = finish - start
print(f'Total Time: {total_time} seconds')
debug = True
