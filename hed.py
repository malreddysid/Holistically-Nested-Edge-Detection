import numpy as np
import tensorflow as tf
import model

class HED():
    def __init__(self, data_loader, num_epochs, chckpnt_dir, weights_path, summary_write_freq, model_save_freq):
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.chckpnt_dir = chckpnt_dir
        self.weights_path = weights_path
        self.summary_write_freq = summary_write_freq
        self.model_save_freq = model_save_freq
        self.num_train = data_loader.get_num_train()

        self.input, self.label, self.loss, self.outputs = model.build_graph()
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # This is needed otherwise, tensorflow assigns the entire GPU memory for one process

        self.saver = tf.train.Saver()

        self.sess = tf.InteractiveSession(config=config)
        self.summary_writer = tf.summary.FileWriter(self.chckpnt_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.load_weights(self.weights_path) # Load weights from VGG model
        ckpt = tf.train.get_checkpoint_state(self.chckpnt_dir) # Continue from previous checkpoint
        if(ckpt and ckpt.model_checkpoint_path):
            saver.restore(sess. ckpt.model_checkpoint_path)


    def train(self):
        self.avg_loss = 0
        total_count = 0
        for ep in range(self.num_epochs):
            self.data_loader.shuffle_data()
            for i  in range(self.num_train):
                self.img, self.lb = self.data_loader.get_data(i)

                val_loss, val_outputs, _ = self.sess.run([self.loss, self.outputs, self.train_op], feed_dict={self.input: self.img, self.label: self.lb})
                self.predictions = self.process_outputs(val_outputs)
                self.avg_loss += val_loss

                if(i == 0 and ep == 0): # Setup summary here so that self.predictions is defined
                    self.summary_op = self.setup_summary()

                if(i % self.summary_write_freq == 0 and i != 0):
                    self.avg_loss = self.avg_loss / self.summary_write_freq
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, total_count)
                    print('Epoch: ' + str(ep) + ',Count: ' + str(total_count) + ',Loss: ' + str(self.avg_loss))
                    total_count = total_count + 1
                    self.avg_loss = 0

                if(i % self.model_save_freq == 0 and i != 0):
                    self.saver.save(self.sess, self.chckpnt_dir + "model.ckpt", total_count)

    def setup_summary(self):
        tf.summary.scalar("avg_loss", self.avg_loss)

        for i in range(len(self.predictions)):
            tf.summary.image('prediction-' + str(i), self.predictions[i])

        tf.summary.image('RGB', self.process_input(self.img))
        tf.summary.image('Label', self.lb * 255)

        summary_op = tf.summary.merge_all()
        return summary_op

    def load_weights(self, weights_path): # Load weights from VGG model
        weights_dict = np.load(weights_path).item()
        for op_name in weights_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in weights_dict[op_name].iteritems():
                    var = tf.get_variable(param_name)
                    self.sess.run(var.assign(data))

    def process_outputs(self, val_outputs): # Process outputs for tensorboard
        for val in val_outputs:
            val = val * 255
        return val_outputs

    def process_input(self, img): # Process inputs for tensorboard
        img[0, :, :, 0] = img[0, :, :, 0] + 122.67892
        img[0, :, :, 1] = img[0, :, :, 1] + 116.66877
        img[0, :, :, 2] = img[0, :, :, 2] + 104.00699
        return img
