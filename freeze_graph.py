import tensorflow as tf

meta_path = 'saved_models/model.ckpt.meta'  # Your .meta file
dir_model = 'saved_models/model.ckpt'
with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess, dir_model)

    # # Output nodes
    # output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node] #all nodes, if you forget which one to add
    # print(output_node_names)
    
    # Freeze the graph

    output_node_names = ['SiameseNN/MatMul_2',
                         'SiameseNN/BiasAdd_2',
                         'SiameseNN/Const_3',
                         'SiameseNN/MatMul_3',
                         'SiameseNN/BiasAdd_3',
                         'SiameseNN/Relu_2',
                         'SiameseNN/Const_4',
                         'SiameseNN/MatMul_4',
                         'SiameseNN/BiasAdd_4',
                         'SiameseNN/Relu_3',
                         'SiameseNN/Const_5',
                         'SiameseNN/MatMul_5',
                         'SiameseNN/BiasAdd_5',
                         'loss']
    
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)

    # Save the frozen graph
    with open('output_graph_full_new.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
