import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    MobileNetV2,
    EfficientNetB4,
    DenseNet121
)
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D, Input, 
    LayerNormalization, MultiHeadAttention, Add, Flatten,
    Reshape, Conv2D
)
from tensorflow.keras.models import Model

def build_cnn_model(base_model_func, num_classes, input_shape=(224, 224, 3)):
    """Builds a standard CNN model using transfer learning."""
    base_model = base_model_func(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False # Freeze base model initially
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def build_vision_transformer(num_classes, input_shape=(224, 224, 3)):
    """Builds a simplified Vision Transformer (ViT) architecture."""
    # This is a basic implementation of a ViT for demonstration
    # In a production environment, one might use keras_cv.models.ViTClassifier
    
    inputs = Input(shape=input_shape)
    
    # Patch extraction using Conv2D (simulating patches)
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) ** 2
    projection_dim = 64
    
    patches = Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )(inputs)
    
    # Reshape to sequence of patches
    encoded_patches = Reshape((num_patches, projection_dim))(patches)
    
    # Transformer block
    for _ in range(2): # 2 Transformer blocks for simplicity/speed
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = Dense(projection_dim * 2, activation="relu")(x3)
        x3 = Dropout(0.1)(x3)
        x3 = Dense(projection_dim)(x3)
        encoded_patches = Add()([x3, x2])
        
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Global average pooling over the sequence
    representation = GlobalAveragePooling1D()(representation)
    
    representation = Dense(128, activation="relu")(representation)
    representation = Dropout(0.5)(representation)
    outputs = Dense(num_classes, activation="softmax")(representation)
    
    return Model(inputs=inputs, outputs=outputs)

def build_hybrid_cnn_transformer(num_classes, input_shape=(224, 224, 3)):
    """Builds the proposed Hybrid CNN-Transformer model."""
    inputs = Input(shape=input_shape)
    
    # Step 1: CNN Feature Extraction (using MobileNetV2 for efficiency)
    cnn_base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    cnn_base.trainable = False
    
    cnn_features = cnn_base.output # Shape: (batch_size, 7, 7, 1280)
    
    # Reshape to sequence for Transformer
    channels = cnn_features.shape[-1]
    
    reshaped_features = Reshape((-1, channels))(cnn_features)
    
    # Projection to smaller dimension for Transformer
    projection_dim = 256
    projected_features = Dense(projection_dim)(reshaped_features)
    
    # Step 2: Transformer Attention Layer
    x1 = LayerNormalization(epsilon=1e-6)(projected_features)
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    x2 = Add()([attention_output, projected_features])
    
    x3 = LayerNormalization(epsilon=1e-6)(x2)
    x3 = Dense(projection_dim * 2, activation="relu")(x3)
    x3 = Dropout(0.1)(x3)
    x3 = Dense(projection_dim)(x3)
    transformer_features = Add()([x3, x2])
    
    # Step 3: Combine and Output
    representation = LayerNormalization(epsilon=1e-6)(transformer_features)
    
    # Global average pooling over the sequence dimension
    representation = GlobalAveragePooling1D()(representation) 
    
    representation = Dense(256, activation="relu")(representation)
    representation = Dropout(0.5)(representation)
    outputs = Dense(num_classes, activation="softmax")(representation)
    
    return Model(inputs=inputs, outputs=outputs)

def get_model(model_name, num_classes):
    models_dict = {
        'VGG16': lambda: build_cnn_model(VGG16, num_classes),
        'ResNet50': lambda: build_cnn_model(ResNet50, num_classes),
        'MobileNetV2': lambda: build_cnn_model(MobileNetV2, num_classes),
        'EfficientNet-B4': lambda: build_cnn_model(EfficientNetB4, num_classes),
        'DenseNet121': lambda: build_cnn_model(DenseNet121, num_classes),
        'Vision Transformer': lambda: build_vision_transformer(num_classes),
        'Hybrid CNN-Transformer': lambda: build_hybrid_cnn_transformer(num_classes)
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models_dict.keys())}")
        
    model = models_dict[model_name]()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
