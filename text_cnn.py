import tensorflow as tf

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, tags_vocab_size, deps_vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_tags = tf.placeholder(tf.int32, [None, sequence_length], name="input_tags")
        self.input_deps = tf.placeholder(tf.int32, [None, sequence_length], name="input_dependency")
        self.input_head = tf.placeholder(tf.int32, [None, sequence_length], name="input_head")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        initializer = tf.contrib.layers.variance_scaling_initializer()

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding_words"):
            self.W = tf.get_variable("embed_W_words", [vocab_size, embedding_size], initializer=initializer)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        with tf.device('/cpu:0'), tf.name_scope("embedding_tags"):
            W_tags = tf.get_variable("embed_W_tags", [tags_vocab_size, embedding_size], initializer=initializer)
            embedded_tags = tf.nn.embedding_lookup(W_tags, self.input_tags)
            embedded_tags_expanded = tf.expand_dims(embedded_tags, -1)

        with tf.device('/cpu:0'), tf.name_scope("embedding_deps"):
            W_deps = tf.get_variable("embed_W_deps", [deps_vocab_size, embedding_size], initializer=initializer)
            embedded_deps = tf.nn.embedding_lookup(W_deps, self.input_deps)
            embedded_deps_expanded = tf.expand_dims(embedded_deps, -1)

        with tf.device('/cpu:0'), tf.name_scope("embedding_head"):
            W_head = tf.get_variable("embed_W_head", [vocab_size, embedding_size], initializer=initializer)
            embedded_head = tf.nn.embedding_lookup(W_head, self.input_head)
            embedded_head_expanded = tf.expand_dims(embedded_head, -1)

        cnn_inputs = tf.concat([self.embedded_chars_expanded, embedded_tags_expanded, embedded_deps_expanded, embedded_head_expanded], -1)
        print("Embedded Shape:", cnn_inputs.shape)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 4, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    cnn_inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply BN
                conv = tf.layers.batch_normalization(conv, axis=-1, training=self.is_training) #axis定的是channel在的维度。
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
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
            self.probabilities = tf.nn.softmax(self.scores)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

