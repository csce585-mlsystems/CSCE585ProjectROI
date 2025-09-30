# Purpose: This file will contain code that helps with Model Development. 

# Body of neccessary imports
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

# end of body of neccessary imports

# Important Steps in Model Development: 1) Obtaining the training data, 2) Create the model containing initialized weights and a bais[which would be very involved with using numbers from certain columns], 3) Observe model's performance before training, 4) Defining a loss function for model, 5) Write a basic Training Loop
# Section 1
# Purpose of section: 1) Obtaining the training data

# end of purpose of section

# Psuedosteps for exercising purpose of section
# - Need to make sure data is clean
# - Need to have an object that references clean dataset
# end of psuedosteps for exercising purpose of section 

# body of section 1



# end of body of section 1
# End of Section 1


# Section 2
# Purpose of section: 2) Create the model containing initialized weights and bias

# end of purpose of section

# Psuedosteps for exercising purpose of section

# end of psuedosteps for exercising purpose of section 

# body of section 2
# NOTE: Code is subject to change, just acting as starting point right now.
class Model(tf.Module):
    def __init__(self):
        # "Randomly genearte weight and bias terms"
        rand_init = tf.random.uniform(shape=[3], minval=0., maxval=5., seed = 22)
        # "Initialize Model Parameters"
        self.w_q = tf.Variable(rand_init[0])
        self.w_l = tf.Variable(rand_init[1])
        self.b = tf.Variable(rand_init[2])
    @tf.function
    def __call__(self,x):
        # "Quadratic Model: quadratic_weight * x^2 + linear_weight*x + bias"
        return self.w_q * (x**2) + self.w_l * x + self.b

# end of body of section 2

# End of Section 2


# Section 3
# Purpose of section:  3) Observe model's performance before training

# end of purpose of section

# Psuedosteps for exercising purpose of section

# end of psuedosteps for exercising purpose of section 

# body of section 3

# NOTE: Code is subject to change, just acting as starting point right now.
quad_model = Model() #<-- instantiation of model
def plot_preds(x,y,f,model,title):
   plt.figure() 
   plt.plot(x,y, '.', label='Data')
   plt.plot(x,f(x), label='Ground Truth')
   plt.plot(x,model(x), label='Predictions')
   plt.title(title)
   plt.legend()

plot_preds(x,y,f,quad_model,'Before Tranining')



# End of body of section 3
# End of Section 3


# Section 4
# Purpose of section: 4) Defining the loss function

# end of purpose of section

# Psuedosteps for exercising purpose of section

# end of psuedosteps for exercising purpose of section 

# End of Section 4


# Section 5
# Purpose of section: 5) Write a basic Training Loop

# end of purpose of section

# Psuedosteps for exercising purpose of section

# end of psuedosteps for exercising purpose of section 

# Body of Section 5
batch_size = 32
# NOTE: This may be a good reference for sending in clean training data for model to use. 
dataset = tf.data.Dataset.from_tensor_slices((x,y))
dataset = dataset.shuffle(buffer_size=x.shape[0].batch(batch)size))

# a) "Set Training Parameters"
epochs = 100
learning_rate = 1e-2
losses= []

# b) "Format Training Loop"
for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            batch_loss = mse_loss(quad_model(x_batch),y_batch) 
        # 1) "Update parameters with respect to the gradient calculations"
        grads = tape.gradient(batch_loss,quad_model.variables)
        for g,v in zip(grads, quad_model.variables):
            v.assign_sub(learning_rate*g)
    # 2) "Keep track of model loss per epoch"
    loss = mse_loss(quad_model(x), y)
    losses.append(loss)
    if epoch %10 == 0:
        print(f'Mean squared error for step {epoch}: {loss.numpy():0.3f}')

# c) "Plot Model Results"
print("\n")
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE loss vs Training Iterations");

plt.show()
# Now, observe your model's performacne after training: 
plot_preds(x,y,f,quad_model, 'After Training')
# End of Body of Section 5
# End of Section 5


