import tensorflow as tf

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, tags_vocab_size, name_vocab_size,
      embedding_size, filter_sizes, filter_sizes2, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_tags = tf.placeholder(tf.int32, [None, sequence_length], name="input_tags")
        self.input_name_entity = tf.placeholder(tf.int32, [None, sequence_length], name="input_name_entities")
        #self.input_deps = tf.placeholder(tf.int32, [None, sequence_length], name="input_dependency")
        #self.input_head = tf.placeholder(tf.int32, [None, sequence_length], name="input_head")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #self.soft_target = tf.placeholder(tf.float32, [None, num_classes], name="soft_target")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.tempreture = tf.placeholder(tf.float32, name="Tempreture")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        #initializer = tf.contrib.layers.variance_scaling_initializer().
        initializer = tf.keras.initializers.he_normal()

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding_words"):
            self.W = tf.get_variable("embed_W_words", [vocab_size, embedding_size], initializer=initializer)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        with tf.device('/cpu:0'), tf.name_scope("embedding_tags"):
            W_tags = tf.get_variable("embed_W_tags", [tags_vocab_size, embedding_size], initializer=initializer)
            embedded_tags = tf.nn.embedding_lookup(W_tags, self.input_tags)
            #embedded_tags_expanded = tf.expand_dims(embedded_tags, -1)

        with tf.device('/cpu:0'), tf.name_scope("embedding_name_entities"):
            W_name = tf.get_variable("embed_W_name", [name_vocab_size, embedding_size], initializer=initializer)
            embedded_name = tf.nn.embedding_lookup(W_name, self.input_name_entity)
            #embedded_name_expanded = tf.expand_dims(embedded_name, -1)

        # with tf.device('/cpu:0'), tf.name_scope("embedding_deps"):
        #     W_deps = tf.get_variable("embed_W_deps", [deps_vocab_size, embedding_size], initializer=initializer)
        #     embedded_deps = tf.nn.embedding_lookup(W_deps, self.input_deps)
        #     embedded_deps_expanded = tf.expand_dims(embedded_deps, -1)
        #
        # with tf.device('/cpu:0'), tf.name_scope("embedding_head"):
        #     W_head = tf.get_variable("embed_W_head", [vocab_size, embedding_size], initializer=initializer)
        #     embedded_head = tf.nn.embedding_lookup(W_head, self.input_head)
        #     embedded_head_expanded = tf.expand_dims(embedded_head, -1)

        cnn_inputs = tf.concat([self.embedded_chars, embedded_tags, embedded_name], -1)
        print("Embedded Shape:", cnn_inputs.shape)

        #################################
        #####   1st Layer  #####
        #################################
        conv1_outs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("first_conv_%s" % filter_size):
                filter_shape = [filter_size, embedding_size * 3, num_filters]
                W = tf.get_variable("first_conv_{}_W".format(filter_size), shape=filter_shape, initializer=initializer)
                conv1 = tf.nn.conv1d(cnn_inputs, W, stride=1, padding="SAME", name="first_conv")
                conv1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training) #axis定的是channel在的维度。
                h = tf.nn.relu(conv1, name="relu_1")
                # h = tf.squeeze(h, axis=2)
                conv1_outs.append(h)

        conv2_inputs = tf.concat(conv1_outs, -1)
        # conv2_inputs = tf.expand_dims(conv2_inputs, -1)
        print("conv2_inputs Shape:", conv2_inputs.shape)

        #################################
        #########   2nd Layer  ##########
        #################################

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes2):
            with tf.variable_scope("second_conv_maxpool_%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, num_filters * len(filter_sizes), num_filters]
                #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                W = tf.get_variable("second_conv_{}_W".format(filter_size), shape=filter_shape, initializer=initializer)
                # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_b_{}".format(filter_size))
                conv = tf.nn.conv1d(
                    conv2_inputs,
                    W,
                    stride=1,
                    padding="VALID",
                    name="second_conv")
                # Apply BN
                conv = tf.layers.batch_normalization(conv, axis=-1, training=self.is_training) #axis定的是channel在的维度。
                # Apply nonlinearity
                h = tf.nn.relu(conv, name="relu_2")
                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, sequence_length - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")

                pooled = tf.layers.max_pooling1d(h, pool_size=sequence_length - filter_size + 1, strides=1, name="pool")

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        print("Conv Shape:", conv.shape)
        print("pooled Shape:", pooled.shape)
        num_filters_total = num_filters * len(filter_sizes2)
        self.h_pool = tf.concat(pooled_outputs, 2)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.probabilities = tf.nn.softmax(self.scores / self.tempreture)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

