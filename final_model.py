## MODEL 

model=Sequential()  

# EMBEDDING LAYER - DISTRIBUTED REPRESENTATION OF TWEETS 
# EMBEDDINGS - GLOVE 100 dimensions further trained on davidson and heot dataset after proper preprocessing

# EMBEDDING DIMENSION = 100
model.add(Embedding(len(vocab)+1, EMBEDDING_DIMENSION, weights=[embedding], input_length=sequence_length, name='embedding_layer'))

# Dropout Layer to reduce overfitting 
model.add(Dropout(0.4))

# LSTM Layer (2 LSTM layers preferable) - Units : 64
model.add(LSTM(64,dropout_W=0.2,dropout_U=0.2))

#Series of dense layers  
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax',name='last'))

# Compiling Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ONE HOT ENCODING - for traing and testing data and transfer learning
Y_train = np_utils.to_categorical(Y_train, num_classes=3)
Y_test = np_utils.to_categorical(Y_test, num_classes=3)
Y_transfer_learning_train=np_utils.to_categorical(Y_transfer_learning_train,num_classes=3)
Y_transfer_learning_test=np_utils.to_categorical(Y_transfer_learning_test,num_classes=3)

# Training Model
model.fit(X_train,Y_train,epochs=25,batch_size=128)

# TRANSFER LEARNING 

# Make previous layers non trainable
for layer in model.layers:
    layer.trainable=False

# Remove last two layers 
model.layers.pop()
model.layers.pop()

# Add two dense layers
model.add(Dense(80, activation='relu'))
model.add(Dense(3,activation='softmax',name='last_transfer_learning'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train model again 
model.fit(X_transfer_learning_train,Y_transfer_learning_train,epochs=25,batch_size=128)