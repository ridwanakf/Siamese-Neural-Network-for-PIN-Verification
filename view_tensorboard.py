import tensorflow as tf

meta_path = 'saved_models/model.ckpt.meta'  # Your .meta file
dir_model = 'saved_models/model.ckpt'
with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess, dir_model)
    writer = tf.summary.FileWriter('./graphs/new', sess.graph)
