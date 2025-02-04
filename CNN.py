from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Assuming X_train is your input matrix and y_train are your labels (0 or 1)
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

#CNNs are excellent for pattern recognition in spatial data. For genomic sequences, CNNs can effectively identify patterns and motifs associated with Helitrons.
#Identifying specific sequence patterns characteristic of Helitrons, which might be missed by traditional sequence alignment tools.