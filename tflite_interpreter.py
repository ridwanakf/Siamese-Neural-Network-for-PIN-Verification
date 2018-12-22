import numpy as np
import tensorflow as tf

test_data1 = np.array(
[0.3160802125930786, 0.7995564341545105, 0.7873384356498718, 1.3119083642959595, 1.3619455099105835, 0.8853291869163513, 0.556852400302887, 0.18962399661540985, 1.1686609983444214, 0.7487136125564575, 0.4847186505794525, 1.5569536685943604, 0.32908371090888977, 1.303735613822937, 1.551056981086731, 1.8119152784347534, 1.2112326622009277, 0.9224334955215454, 1.2264668941497803, 0.6652265191078186, 2.376328945159912, 0.170978844165802, 0.6423753499984741, 0.20936952531337738, 0.238, 0.213, 0.204, 0.264, 0.303, 0.233, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01568627543747425, 0.011764707043766975, 0.011764707043766975, 0.01568627543747425, 0.02352941408753395, 0.027450982481241226, 0.03921568766236305, 0.03529411926865578, 0.06666667014360428, 0.05098039656877518, 0.03529411926865578, 0.05490196496248245], dtype=np.float32)

test_data2 = np.array(
[1.3689281940460205, 1.0927664041519165, 2.6036016941070557, 3.246880531311035, 2.032926559448242, 0.9140530228614807, 1.394978642463684, 1.0679513216018677, 1.28534996509552, 0.19055487215518951, 0.38479816913604736, 0.2899835705757141, 2.360123872756958, 0, 4.228667736053467, 5.013617992401123, 2.8996362686157227, 0.7473423480987549, 1.9745728969573975, 2.3957531452178955, 2.276880979537964, 0.7797783017158508, 0.4287991225719452, 0.642052173614502, 0.163, 0.178, 0.233, 0.291, 0.293, 0.172, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.007843137718737125, 0.01568627543747425, 0.0313725508749485, 0.007843137718737125, 0.01568627543747425, 0.011764707043766975, 0.05490196496248245, 0.062745101749897, 0.04313725605607033, 0.04313725605607033, 0.05490196496248245, 0.0470588281750679]

, dtype=np.float32)


# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="converted_model_new.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
# input_shape1 = input_details[0]['shape']
# input_shape2 = input_details[1]['shape']
# print('input shape1 =', input_shape1)
# print('input shape2 =', input_shape2)

# change the following line to feed into your own data.

# input_data1 = np.array(np.random.random_sample(input_shape1), dtype=np.float32)
# input_data2 = np.array(np.random.random_sample(input_shape1), dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data1)
# interpreter.set_tensor(input_details[1]['index'], input_data2)


test_data1 = test_data1.reshape(input_details[0]['shape'])
test_data2 = test_data2.reshape(input_details[0]['shape'])


interpreter.set_tensor(input_details[0]['index'], test_data1)
interpreter.set_tensor(input_details[1]['index'], test_data2)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index']) # 0 = index outputnya (bisa aja output lebih dari 1)
print(output_data)
