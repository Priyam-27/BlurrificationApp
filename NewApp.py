import cv2
import cvlib
import numpy as np
from PIL import Image
import streamlit as st
import tempfile
# import pytesseract
# from pytesseract import Output
from nudenet import NudeDetector

eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# word = ['what', 'want', 'You', 'you', 'know', "touch", 'those']        # list of words to be masked

detector = NudeDetector()
def nudity_blur(img):
	classes = ['EXPOSED_ANUS', 'EXPOSED_BUTTOCKS', 'COVERED_BREAST_F', 'EXPOSED_BREAST_F',
           'EXPOSED_GENITALIA_F', 'EXPOSED_GENITALIA_M', 'EXPOSED_BUTTOCKS', 'EXPOSED_BREAST_F', 'EXPOSED_GENITALIA_F',
           'EXPOSED_GENITALIA_M', 'EXPOSED_BREAST_M']
	# detector = ND()

	for i in detector.detect(img):
		if i['label'] in classes:
#             if i['label'] in []
			x,y,w,h = i['box']
			Img = cv2.medianBlur(img[y:h, x:w], ksize=151)
			img[y:h, x:w] = Img
	return img


# def remove_punc(word):
#     punc = [',', '.', '-', '/', '@', '"']
#     for ele in word:  
#         if ele in punc:  
#             word = word.replace(ele, "") 
#     return word

# def word_coor(image):
#     boxes=[]
#     texts=[]
#     rgb= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # converts image to rgb
#     results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
#     if set(results['text']) == '':
#         return ('No text detected!!!')
#     else:
#         for i in range(0,len(results['top'])):  ## iterating through all the values for the words present
#             text = remove_punc(results['text'][i])          ## returning the text value read by OCR
#             conf = int(results['conf'][i])          ## returning confidence of word read by OCR
            
#             x= results['left'][i]              ## top left x coordinates of word read
#             y= results['top'][i]               ## top left y coordinate of word read
#             w= results['width'][i]             ## width of word read
#             h= results['height'][i]            ## height of word read
            
#             if conf > 50:
#                 boxes.append([x,y,w,h])
#                 texts.append(text) 
#         return list(zip(texts, boxes))
        
# def word_mask(img, list):
#     for word, coor in word_coor(img):
#         if word in list:
#             x,y,w,h = coor
#             roi = img[y:y+h, x:w+x]
#             img[y:y+h, x:w+x] = cv2.medianBlur(roi, ksize=151)
#     return img

def face_blur(img):
	coor, _ = cvlib.detect_face(img)
	for face in coor:
		x,y,w,h = face
		roi = img[y:h, x:w]
		img[y:h, x:w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_eyes(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eyes=eye_classifier.detectMultiScale(gray, 1.3,5)
	if eyes == ():
		# return img
		print('No Eyes Detected')
	else:
		for x,y,w,h in eyes:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_eyes_video(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eyes=eye_classifier.detectMultiScale(gray, 1.6,5)
	if eyes == ():
		# return img
		print('No Eyes Detected')
	else:
		for x,y,w,h in eyes:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_smile(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile=smile_classifier.detectMultiScale(gray, 1.6,8)
	if smile == ():
		# return img
		print('No Smile Detected')
	else:
		for x,y,w,h in smile:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_smile_video(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile=smile_classifier.detectMultiScale(gray, 1.5,10)
	if smile == ():
		# return img
		print('No Smile Detected')
	else:
		for x,y,w,h in smile:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img


## Deploying the App


def about():
	st.markdown('''This Web App is made for censoring(blurring) the NSWF(Not Safe For Work) materials, that includes personal pictures depicting Nudity.
		This App will blur the inappropriate areas and body. Along with that we have provided option for blurrring Eyes, Smile, Face and Nudity.''')

	st.markdown('''YOLO custom object dedection is used for detection of Nudity whereas HAAR Cascade Classifier is used for detecting Face, Smile and Eyes.''')


def main():
	st.title("Object Detection and Masking")
	st.subheader('For Recorded as well as Real-time media')
	st.write('Using YOLOV3 object detection and Haar Cascade Classifier we detect the NSWF parts and blur them with OpenCV')

	activities = ['Home', 'About']
	choice = st.sidebar.selectbox('Select an option', activities)

	if choice == 'Home':
		st.write('Go to the about section to know more about it')

		file_type = ['Image', 'Video']
		file_choice = st.sidebar.radio('Select file type', file_type)

		if file_choice == 'Video':
			file = st.file_uploader('Choose file', ['mp4'])
			tfile = tempfile.NamedTemporaryFile(delete=False)
			tfile.write(file.read())

			choice_type = st.sidebar.radio('Make your choice', ['Original', 'Eyes', 'Face', 'Smile', 'Nudity'])

			if st.button('Process'):
				if choice_type == 'Original':
					vf = cv2.VideoCapture(tfile.name)
					stframe = st.empty()

					while vf.isOpened():
						ret, frame = vf.read()
    			# if frame is read correctly ret is True
						if not ret:
							print("Can't receive frame (stream end?). Exiting ...")
							break
						# if frame == None:
						# 	pass
						frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						stframe.image(frame)

				elif choice_type == 'Eyes':
					vf = cv2.VideoCapture(tfile.name)
					stframe = st.empty()

					while vf.isOpened():
						ret, frame = vf.read()
    			# if frame is read correctly ret is True
						if not ret:
							print("Can't receive frame (stream end?). Exiting ...")
							break
						# if frame == None:
						# 	pass
						frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						stframe.image(blur_eyes_video(frame))


				elif choice_type == 'Face':
					vf = cv2.VideoCapture(tfile.name)
					stframe = st.empty()

					while vf.isOpened():
						ret, frame = vf.read()
    			# if frame is read correctly ret is True
						if not ret:
							print("Can't receive frame (stream end?). Exiting ...")
							break
						# if frame == None:
						# 	pass
						frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						stframe.image(face_blur(frame))


				elif choice_type == 'Smile':
					vf = cv2.VideoCapture(tfile.name)
					stframe = st.empty()

					while vf.isOpened():
						ret, frame = vf.read()
    			# if frame is read correctly ret is True
						if not ret:
							print("Can't receive frame (stream end?). Exiting ...")
							break
						# if frame == None:
						# 	pass
						frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						stframe.image(blur_smile_video(frame))



				elif choice_type == 'Nudity':
					vf = cv2.VideoCapture(tfile.name)
					stframe = st.empty()

					while vf.isOpened():
						ret, frame = vf.read()
    			# if frame is read correctly ret is True
						if not ret:
							print("Can't receive frame (stream end?). Exiting ...")
							break
						# if frame == None:
						# 	pass
						frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						image =nudity_blur(frame)
						stframe.image(image)
						# if len(label) == 0:
						# 	st.info('No Nudity present in the video')

				# elif choice_type == 'Text':
				# 	vf = cv2.VideoCapture(tfile.name)
				# 	stframe = st.empty()

				# 	while vf.isOpened():
				# 		ret, frame = vf.read()
    # 			# if frame is read correctly ret is True
				# 		if not ret:
				# 			print("Can't receive frame (stream end?). Exiting ...")
				# 			break
				# 		# if frame == None:
				# 		# 	pass
				# 		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				# 		stframe.image(word_mask(frame, word))



		elif file_choice == 'Image':

			image_file=st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png', 'webp'])

			if image_file is not None:
				image = Image.open(image_file)
				image = np.array(image)

				choice_type = st.sidebar.radio('Make a choice', ('Original','Eyes', 'Face', 'Smile', 'Nudity'))

				if st.button('Process'):
					if choice_type == 'Original':
						result_image = image
						st.image(result_image, use_column_width=True)
						st.info(f'{choice_type} image returned')

					
					elif choice_type == 'Eyes':
						result_image= blur_eyes(image)
						st.image(result_image, use_column_width=True)
						st.info(f'{choice_type} of the image got blurred')
						if blur_eyes(image) == 'No Eyes Detected':
							st.info('No Eyes Detected')
					# st.info(blur_eyes(image))	
					elif choice_type == 'Face':
						result_image= face_blur(image)
						st.image(result_image, use_column_width=True)
						st.info(f'{choice_type} of the image got blurred')
						if face_blur(image) == 'No Face Detected':
							st.info('No Face Detected')
					# st.info(blur_face(image))
					elif choice_type == 'Smile':
						result_image= blur_smile(image)
						st.image(result_image, use_column_width=True)
						st.info(f'{choice_type} of the image got blurred')
						if blur_smile(image) == 'No Smile Detected':
							st.info('No Smile Detected')
					# st.info(blur_smile(image))
					elif choice_type == 'Nudity':
						result_image= nudity_blur(image)
						st.image(result_image, use_column_width=True)


					# elif choice_type == 'Text':
					# 	result_image = word_mask(image, word)
					# 	st.image(result_image, use_column_width=True)


	elif choice =='About':
		about()

if __name__ == '__main__':
	main()



