import tensorflow as tf

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_inputs = tf.keras.layers.RandomContrast(factor=0.1, seed=seed)
    self.augment_labels = tf.keras.layers.RandomContrast(factor=0.1, seed=seed)
    self.augment_inputs = tf.keras.layers.RandomRotation(factor=0.1, seed=seed)
    self.augment_labels = tf.keras.layers.RandomRotation(factor=0.1, seed=seed)
    self.augment_inputs = tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=seed)
    self.augment_labels = tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=seed)
    self.augment_inputs = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=seed)
    self.augment_labels = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=seed)
    self.augment_inputs = tf.keras.layers.ElasticDistortion(alpha=1, sigma=0.07, seed=seed)
    self.augment_labels = tf.keras.layers.ElasticDistortion(alpha=1, sigma=0.07, seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels