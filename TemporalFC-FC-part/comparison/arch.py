# def build_model(self):
#     """
#     Build and compile model
#     :return:
#     """
#     s, p, o = tf.keras.Input(1, name='Subject'), tf.keras.Input(1, name='Predicate'), tf.keras.Input(1, name='Object')
#
#     ent_layer = tf.keras.layers.Embedding(
#         self.ent_no,
#         self.dims,
#         embeddings_initializer=tf.keras.initializers.Constant(ent_matrix),
#         trainable=False,
#         name='Entity_embeddings'
#     )
#
#     rel_layer = tf.keras.layers.Embedding(
#         self.rel_no,
#         self.dims,
#         embeddings_initializer=tf.keras.initializers.Constant(rel_matrix),
#         trainable=False,
#         name='Relation_embeddings'
#     )
#
#     sp_merge = tf.keras.layers.Concatenate(axis=1, name='SP_Merge')([ent_layer(s), rel_layer(p)])
#     po_merge = tf.keras.layers.Concatenate(axis=1, name='PO_Merge')([rel_layer(p), ent_layer(o)])
#
#     dense_layer = tf.keras.layers.Dense(self.dims, activation='relu')(sp_merge)
#     drop_layer = tf.keras.layers.Dropout(.2, name='Dropout_1')(dense_layer)
#
#     dense_layer2 = tf.keras.layers.Dense(self.dims, activation='relu')(po_merge)
#     drop_layer2 = tf.keras.layers.Dropout(.2, name='Dropout_2')(dense_layer2)
#
#     merge = tf.keras.layers.Concatenate(axis=1, name='Merge')([drop_layer, drop_layer2])
#
#     dense_layer3 = tf.keras.layers.Dense(self.dims, activation='relu')(merge)
#     drop_layer3 = tf.keras.layers.Dropout(.2, name='Dropout_3')(dense_layer3)
#
#     flatten = tf.keras.layers.Flatten()(drop_layer3)
#
#     output_layer = tf.keras.layers.Dense(self.ent_no, activation='sigmoid', name='Output_layer_tail')(flatten)
#
#     output_layer2 = tf.keras.layers.Dense(self.ent_no, activation='sigmoid', name='Output_layer_head')(flatten)
#
#     model = tf.keras.models.Model(inputs=(s, p, o), outputs=[output_layer, output_layer2])
#     bce = tf.keras.losses.BinaryCrossentropy()
#     opt = tf.keras.optimizers.Adam(self.lr)
#
#     model.compile(loss=bce, optimizer=opt)
#
#     print(model.summary())
#     return model
