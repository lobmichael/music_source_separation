import tensorflow as tf

def Downsampling(x, filters, kernel_size = (5,5), padding = 'same', stride = 2, multires = False):
    '''
    Downsampling Block
    
    Arguments:
        x : input layer (tf.keras.layer)
        filters : number of filters (int)
        kernel_size : kernel dimensions (tuple or int), default (5,5)
        padding : padding type for convolution (string), default same
        stride : stride for convolution (tuple or int), default 2
    
    Returns:
        output : output layer (tf.keras.layer)
    '''
    if multires == False:
      conv = tf.keras.layers.Conv2D(kernel_size = kernel_size, filters = filters, strides = stride, padding = padding,data_format = "channels_last")(x)
    elif multires == True:
      conv = tf.keras.layers.Conv2D(kernel_size = kernel_size, filters = filters//2, strides = stride, padding = padding,data_format = "channels_last")(x)
      conv3 = tf.keras.layers.Conv2D(kernel_size = (3,3), filters = filters//4, strides = stride, padding = padding,data_format = "channels_last")(x)
      conv7 = tf.keras.layers.Conv2D(kernel_size = (7,7), filters = filters//4, strides = stride, padding = padding,data_format = "channels_last")(x)
      conv = tf.keras.layers.Concatenate()([conv, conv3, conv7])
    bn = tf.keras.layers.BatchNormalization()(conv)
    output = tf.keras.layers.LeakyReLU(0.2)(bn)

    return output

def Upsampling(x , y, filters, res_filts, kernel_size = (5,5), padding = 'same', stride = 2, dropout = 'False', resblock = True, se_block = False):
    '''
    Upsampling Block
    
    Arguments:
        x : input layer (tf.keras.layer)
        y : residual connection layer (tf.keras.layer)
        filters : number of filters (int)
        kernel_size : kernel dimensions (tuple or int), default (5,5)
        padding : padding type for convolution (string), default same
        stride : stride for convolution (tuple or int), default 2
        dropout : dropout (boolean), default False
    
    Returns:
        output : output layer (tf.keras.layer)
    '''
    
    conv = tf.keras.layers.Conv2DTranspose(kernel_size = kernel_size, filters = filters, strides = stride, padding = padding, data_format = "channels_last")(x)
    act = tf.keras.layers.ReLU()(conv)
    output = tf.keras.layers.BatchNormalization()(act)
    if dropout == 'True':
      output = tf.keras.layers.Dropout(0.5)(output)
    if y is not None:
      if resblock is True:
        y = ResBlock(y, depth = 2, filters = res_filts)
      output = tf.keras.layers.Concatenate()([y, output])
    if se_block is True:
      output = SE_Block(output, r = 16)
    return output

def ResBlock(x, filters, depth = 2, kernel_size = (5,5), padding = 'same', method = 'concat', se_block = False):
      '''
    ResNet Block
    
    Arguments:
        x : input layer (tf.keras.layer)
        depth : number of layers in ResBlock
        filters : number of filters (int)
        kernel_size : kernel dimensions (tuple or int), default (5,5)
        padding : padding type for convolution (string), default same
        dropout : dropout (boolean), default False
    
    Returns:
        output : output layer (tf.keras.layer)
    '''

      conv = tf.keras.layers.Conv2D(kernel_size = kernel_size, filters = filters, padding = padding, data_format = "channels_last")(x)
      conv = tf.keras.layers.ReLU()(conv)
      conv = tf.keras.layers.BatchNormalization()(conv)
      for i in range(0,depth-1):
        conv = tf.keras.layers.Conv2D(kernel_size = kernel_size, filters = filters, padding = padding, data_format = "channels_last")(conv)
        conv = tf.keras.layers.ReLU()(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
      if method == 'add':
        output = tf.keras.layers.Add()([x, conv])
      elif method == 'concat':
        output = tf.keras.layers.Concatenate()([x, conv])
      
      output = tf.keras.layers.ReLU()(output)

      if se_block is True:
       output = SE_Block(output, r = 16)

      return output

def SE_Block(x, r = 16):

  '''
    Squeeze and Excitation Block
    Assumes channel_last format
    
    Arguments:
        x : input layer (tf.keras.layer)
        r : reduction ratio for first FC layer
    
    Returns:
        output : output layer (tf.keras.layer)
  '''
  filters = x.shape[-1]
  pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)
  fc1 = tf.keras.layers.Dense(int(filters/r))(pool)
  fc1 = tf.keras.layers.ReLU()(fc1)
  fc2 = tf.keras.layers.Dense(filters)(fc1)
  fc2 = tf.keras.layers.Activation('sigmoid')(fc2)
  output = tf.keras.layers.Reshape([1,1,filters])(fc2)

  output = tf.keras.layers.Multiply()([x,output])

  return output

def Steminator(input_shape = (256,128,1), kernel_size = (5,5), feature_maps = 8, multires = True, resblock = True, se_block = True):

    '''
    MultiResUnet Network Builder - Steminator
    
    Arguments:
        input_shape : input shape (tuple)
        depth : number of layers in ResBlock
        feature_maps : number of initial filters (int)
        kernel_size : kernel dimensions (tuple or int), default (5,5)
        multires : use multi-res Unet (boolean), default True
        resblock : use resblock residual connections (boolean), default True
    
    Returns:
        model : tf.keras Neural net model (tf.keras.Model)
    '''

    cqt_input = tf.keras.Input(shape=input_shape)
    
    ds_0 = Downsampling(cqt_input, filters = feature_maps*2, multires = multires)
    ds_1 = Downsampling(ds_0, filters = feature_maps*4, multires = multires)
    ds_2 = Downsampling(ds_1, filters = feature_maps*8, multires = multires)
    ds_3 = Downsampling(ds_2, filters = feature_maps*16, multires = multires)
    ds_4 = Downsampling(ds_3, filters = feature_maps*32, multires = multires)
    ds_5 = Downsampling(ds_4, filters = feature_maps*64, multires = multires)
    
    us_0 = Upsampling(ds_5,ds_4,filters = feature_maps*32, res_filts = feature_maps, dropout = 'True', resblock = resblock)
    us_1 = Upsampling(us_0,ds_3,filters = feature_maps*16, res_filts = feature_maps*2, dropout = 'True', resblock = resblock)
    us_2 = Upsampling(us_1,ds_2,filters = feature_maps*8, res_filts = feature_maps*4, dropout = 'True', resblock = resblock)
    us_3 = Upsampling(us_2,ds_1,filters = feature_maps*4, res_filts = feature_maps*8, resblock = resblock)
    us_4 = Upsampling(us_3,ds_0,filters = feature_maps*2, res_filts = feature_maps*16, resblock = resblock, se_block = False)
    us_5 = Upsampling(us_4,None,filters = feature_maps, res_filts = feature_maps*32, resblock = resblock, se_block = False)

    
    mask = tf.keras.layers.Conv2D(kernel_size = (1,1), filters = 1,activation='relu', padding = 'same',data_format="channels_last")(us_5) #original network kernel_size = (1,1)

    outputs = tf.keras.layers.Multiply()([cqt_input,mask])
  
    model = tf.keras.Model(inputs = cqt_input, outputs = outputs, name='Steminator')

    #model.summary()

    return model