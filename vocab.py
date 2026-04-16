

"""
Handels the word to vector to word process
To do list:    
    1. create a word to vec dictionary
    2. create a vec to word dictionary 
    3. create a frequency threshold: (if a word appears less that a certain number of times, don't bother training the model on it)
    4. Figure out how to encorporate the tokens into the captions

"""
#import the system settings from the config file 
import config
import re

class Vocabulary: 

    def __init__(self): 

        #initializes the dictionaries as empty 
        self.word_to_idx = {}
        self.idx_to_word = {}

        #set current index to zero 
        self.curr_idx = 0

        #adds the tokens to the dictionaries before other words go in
        self.add_word(config.PAD_TOKEN)
        self.add_word(config.START_TOKEN)
        self.add_word(config.END_TOKEN)
        self.add_word(config.UNK_TOKEN)

    #define how to add a word to the vocabulary
    def add_word(self, word): 

        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.curr_idx
            self.idx_to_word[self.curr_idx] = word
            self.curr_idx += 1
        
    def build_vocab(self, captions): 
        frequency = {}

        #tests if a word is new or not, if new, set freq to 1, if not, increments the frequency 
        for caption in captions: 
            clean_caption = re.sub(r'[^a-zA-Z\s]', '', caption.lower())
            for word in clean_caption.lower().split(): #sets the words as lowercase and splits them by the spaces. 
                if word not in frequency: 
                    frequency[word] = 1
                else: 
                    frequency[word] += 1

        for word in frequency: 
            if frequency[word] >= config.THRESHOLD: 
                self.add_word(word)

    def caption_to_idx(self, caption): 
        #will store the indices in a list 
        indices = [self.word_to_idx[config.START_TOKEN]] #starts w/ the start token 

        clean_caption = re.sub(r'[^a-zA-Z\s]', '', caption.lower())
      

        #goes through each word in the caption and assigns the respective index (if the word has not be learend, it assigns an unknown tokne)
        for word in clean_caption.lower().split():
            if word in self.word_to_idx: 
                indices.append(self.word_to_idx[word])
            else: 
                indices.append(self.word_to_idx[config.UNK_TOKEN])
            
        #adds the end token to the index list 
        indices.append(self.word_to_idx[config.END_TOKEN])

        return indices

    def idx_to_captions(self,indices):   
        #create a list to store the words 
        caption = [] 

        for idx in indices:
            word = self.idx_to_word[idx]
            if word in [config.START_TOKEN, config.PAD_TOKEN, config.UNK_TOKEN]: #skips over the non break tokens
                continue 
            elif word == config.END_TOKEN: #stops the loop if it hits the break token
                break
            else: 
                caption.append(self.idx_to_word[idx]) #adds the word to the caption list

        return " ".join(caption) #combines the list of words into a string
