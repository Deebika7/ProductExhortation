# Important imports
from app import app
from flask import request, render_template, url_for
import os
import cv2
import numpy as np
from PIL import Image
import random
import string
import pytesseract

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():

	# Execute if request is get
	if request.method == "GET":
		full_filename =  'images/white_bg.jpg'
		return render_template("index.html", full_filename = full_filename)

	# Execute if reuqest is post
	if request.method == "POST":
		image_upload = request.files['image_upload']
		imagename = image_upload.filename
		image = Image.open(image_upload)

		# Converting image to array
		image_arr = np.array(image.convert('RGB'))
		# Converting image to grayscale
		gray_img_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
		#Converting image back to rbg
		image = Image.fromarray(gray_img_arr)

		# Printing lowercase
		letters = string.ascii_lowercase
		# Generating unique image name for dynamic image display
		name = ''.join(random.choice(letters) for i in range(10)) + '.png'
		full_filename =  'uploads/' + name

		# Extracting text from image
		custom_config = r'-l eng --oem 3 --psm 6'
		text = pytesseract.image_to_string(image,config=custom_config)

		# Remove symbol if any
		characters_to_remove = "!()@—*“>+-/,'`|£#%$&^_~"
		new_string = text
		for character in characters_to_remove:
			new_string = new_string.replace(character, "")

		# Converting string into list to dislay extracted text in seperate line
		new_string = new_string.split("\n")
        #here

		def func(name,lis):
			for i in lis:
				if i.lower().__contains__(name):
					i=i.lower()
					index=i.rindex(name)
					res="" 
					f=1
					#for j,k in zip (i[index::],i[index+1::]):
					for j in i[index::]:
						if((j.isdigit()) & f==1):
							res+=j
							n=i.index(j)
							
							if(n+1<len(i)):
								if (not i[n+1].isdigit()):
									f=0
					return res
			return 0
		#list=['energy g 233 carb g 56','sugar g 34']
		l=[]
		if(func('energy',new_string)!=0):
			l.append(func('energy',new_string))
		else:
			l.append(func('calories',new_string))
		l.append(func('protein',new_string))
		l.append(func('carbohydrate',new_string))
		l.append(func('sugar',new_string))
		if(func('fat',new_string)!=0):
			l.append(func('fat',new_string))
		else:
			l.append(func('total fat',new_string))
		l.append(func('transfat',new_string))
		l.append(func('saturated fat',new_string))
        #here
		# Saving image to display in html
		img = Image.fromarray(image_arr, 'RGB')
		img.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))
		# Returning template, filename, extracted text
		return render_template('index.html', full_filename = full_filename, text = l)

# Main function
if __name__ == '__main__':
    app.run(debug=True)
