import tensorflow as tf
import numpy as np
from kbody import inference
from kbody_transform import MultiTransformer
from kbody_input import extract_xyz

batch_inputs = tf.placeholder(tf.float32, shape=[None, 1, None, 3], name="batch_input")
batch_occurs = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name="batch_occurs")
batch_weights = tf.placeholder(tf.float32, shape=[None, 1, None, 1], name="batch_weights")
batch_split_dims = tf.placeholder(tf.int64, shape=[1, ], name="batch_split_dims")
nat = 1
kbody_terms = ["BBB"]
conv_sizes = (80, 180, 120, 60, 40)


y_total, _ = inference(
  batch_inputs, batch_occurs, batch_weights, batch_split_dims, nat, kbody_terms, conv_sizes=conv_sizes)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

saver = tf.train.Saver()
saver.restore(sess, "./models/Bx-.v4/model.ckpt-167273")

clf = MultiTransformer(["B"])

array_of_species, y_true, coords, _ = extract_xyz("/Users/bismarrck/Downloads/Bx/B39-_opted.xyz", 1000, 39)


graph = tf.get_default_graph()

kernel = graph.get_tensor_by_name("one-body/weights:0")

print(kernel.eval())

abs_diff = []
raw_diff = []
preds = []

for i in range(100):

  features, splits, target, weights, occurs = clf.transform(array_of_species[i], coords[i], y_true[i])

  features = features.reshape((1, 1, -1, 3))
  weights = weights.reshape((1, 1, -1, 1))
  occurs = occurs.reshape((1, 1, 1, 1))

  splits = np.array(splits, dtype=np.int32)
  y_pred = sess.run(
    y_total,
    feed_dict={
      batch_inputs: features,
      batch_weights: weights,
      batch_occurs: occurs,
      batch_split_dims: splits}
  )

  d = -y_pred - y_true[i]
  raw_diff.append(d)
  abs_diff.append(np.abs(d))

  preds.append(-y_pred)


preds = np.array(preds)
abs_diff = np.array(abs_diff)
raw_diff = np.array(raw_diff)

print("abs mean error: ", np.mean(abs_diff))
print("raw mean error: ", np.mean(raw_diff))

relative_pred = preds - preds[0]
relative_true = y_true[:100] - y_true[0]


print(np.mean(relative_pred), np.mean(relative_true))
print(np.std(relative_pred), np.std(relative_true))

print('')
