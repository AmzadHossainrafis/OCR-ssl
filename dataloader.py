
from utils import * 
config=read_yaml()

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self,batch_size,data,labels,characters,CTC) :
        self.batch_size = batch_size
        self.dataS = data
        self.labels = labels
        self.characters = characters
        self.CTC=CTC # if you use build_model() function then pass True to CTC argument


    def __data_preprocess(self,images, labels,characters):
        """
        Preprocess the images and labels.
        :param images: The images.
        :param labels: The labels.

        :return: The preprocessed images and labels.
        
            """
        
        
        char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
        img = tf.io.read_file(images)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=config["chennel"])
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [config['height'], config['width']])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = char_to_num(tf.strings.unicode_split(labels, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return img,label

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataS) / float(self.batch_size)))

    def __getitem__(self, idx) -> tuple:
        batch_x = self.dataS[idx * self.batch_size:(idx + 1) * self.batch_size]  # get the batch of images
        batch_lable = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size] # get the batch of labels

        img=np.zeros((self.batch_size,200,50,1)) # create a numpy array of zeros to hold the images
        lab=np.zeros((self.batch_size,5))
 
        #fro loop for img and lables 
        for i in range(len(batch_x)):
            imgs,lable=self.__data_preprocess(batch_x[i],batch_lable[i],self.characters)
            img[i]=imgs
            lab[i]=lable
            
        if self.CTC:
            return ([img,lab],None)# this model take 2 input(img and lab) at once for ctc loss claculation 

        else:
            return img, lab #if the ctc loss is passed in copiler as loss
