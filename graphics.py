import pygame
import pygame.camera
import pickle
import numpy as np
import sys
from tkinter import Tk
from tkinter import filedialog
from PIL import Image


classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise" ]
netPath = "C:\\Users\\user\\Documents\\SCHOOL\\DS\\project\\saved_modules2\\adam7"
picButton = (400,700)
uploadButton = (400,820)
WHITE = (255,255,255)
BLACK = (0, 0, 0)

def openPickle(path):
    f = open(path,"rb")
    return pickle.load(f)

def prepare(img): #resize+grayscale - 1X48X48X1
    subsurface = img.subsurface(pygame.Rect(80, 0, (img.get_width() - 80), img.get_height()))
    img = pygame.transform.scale(subsurface, (48, 48)) #transforming from 480X480 - > 48X48
    imgArr = pygame.surfarray.array3d(img)
    imgArr = imgArr.swapaxes(0, 1)
    imgArr = np.dot(imgArr,[0.2989, 0.5870, 0.1140]) #grayscaling
    imgArr= imgArr.reshape(1,48,48,1)
    return imgArr/255 #normalizing

net = openPickle(netPath)

pygame.init()

REFRESH_RATE=60
clock = pygame.time.Clock()

pygame.camera.init()
cameras = pygame.camera.list_cameras()
webcam = pygame.camera.Camera(cameras[0])
webcam.start()
img = webcam.get_image()

pygame.font.init()
myfont = pygame.font.SysFont('Arial', 30)
myfont_percent = pygame.font.SysFont('Arial', 20)

screen = pygame.display.set_mode( ( img.get_width()+400, img.get_height()+450 ) )
pygame.display.set_caption("Emotion Camera")
take_pic = 0
pygame.draw.rect(screen, WHITE,(picButton[0], picButton[1],200,100))
textsurface = myfont.render('Take Picture', False, (0, 0, 0))
screen.blit(textsurface,(picButton[0]+30,picButton[1]+30))

pygame.draw.rect(screen, WHITE,(uploadButton[0], uploadButton[1],200,100))
textsurface2 = myfont.render('Upload Picture', False, (0, 0, 0))
screen.blit(textsurface2,(uploadButton[0]+20,uploadButton[1]+30))

while True :
    for e in pygame.event.get() :
        if e.type == pygame.QUIT :
            pygame.quit()
            sys.exit()
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1: #left click
            x,y = pygame.mouse.get_pos()
            if(x>picButton[0] and y>picButton[1] and x<picButton[0]+200 and y<picButton[1]+100): #on button
                if (take_pic == 0):
                    print("picture captured")
                    take_pic = 1
                    pygame.draw.rect(screen, WHITE, (picButton[0], picButton[1], 200, 100))
                    textsurface = myfont.render('Retake', False, BLACK)
                    screen.blit(textsurface, (picButton[0] + 60, picButton[1] + 30))
                    screen.blit(img, (200, 200)) #displaying img that is being predicted
                    image = prepare(img) #preparing image
                    prediction = net.predict(image) #predict
                    percentages = prediction[1][0] * 100 / np.sum(prediction[1][0]) #turning output (axis 1) into percentages.
                    prediction = classes[int(prediction[0])] #highest value - prediction
                    textsurface = myfont.render(str('DETECTED EMOTION: ' + prediction), False, WHITE)
                    screen.blit(textsurface, (350, 100)) #printing prediction
                    #printing percentages
                    for c in range(len(classes)):
                        string = str(classes[c]) + ": " + str(round(percentages[c], 2)) + "%"
                        print(string)
                        textsurface1 = myfont_percent.render(string, False, WHITE)
                        screen.blit(textsurface1, (20, 280 + 50 * c))
                    pygame.display.flip()
                elif (take_pic==1):
                    print("restarting camera")
                    screen.fill(BLACK)
                    pygame.draw.rect(screen, WHITE, (uploadButton[0], uploadButton[1], 200, 100))
                    textsurface2 = myfont.render('Upload Picture', False, (0, 0, 0))
                    screen.blit(textsurface2, (uploadButton[0] + 20, uploadButton[1] + 30))
                    pygame.draw.rect(screen, WHITE, (picButton[0], picButton[1], 200, 100))
                    textsurface = myfont.render('Take Picture', False, (0, 0, 0))
                    screen.blit(textsurface, (picButton[0] + 35, picButton[1] + 30))
                    take_pic =0
            if (x > uploadButton[0] and y > uploadButton[1] and x < uploadButton[0] + 200 and y < uploadButton[1] + 100):  # on button
                print("upload a picture")
                screen.fill(BLACK)
                pygame.draw.rect(screen, WHITE, (uploadButton[0], uploadButton[1], 200, 100))
                textsurface2 = myfont.render('Upload Picture', False, (0, 0, 0))
                screen.blit(textsurface2, (uploadButton[0] + 20, uploadButton[1] + 30))
                pygame.draw.rect(screen, WHITE, (picButton[0], picButton[1], 200, 100))
                textsurface = myfont.render('Retake', False, BLACK)
                screen.blit(textsurface, (picButton[0] + 60, picButton[1] + 30))
                take_pic = 1
                Tk().withdraw()
                filename = filedialog.askopenfilename()
                print('Selected:', filename)
                if filename != "":
                    img = pygame.transform.scale(pygame.image.load(filename), (480, 480))
                    screen.blit(img, (280, 200))
                    pygame.display.flip()
                    image = np.asarray(Image.open(filename)).reshape(1, 48, 48, 1)/255
                    prediction = net.predict(image)
                    percentages = prediction[1][0]*100/np.sum(prediction[1][0])
                    prediction = classes[int(prediction[0])]
                    textsurface = myfont.render(str('DETECTED EMOTION: ' + prediction), False, WHITE)
                    screen.blit(textsurface, (350, 100))
                    for c in range(len(classes)):
                        string = str(classes[c])+": "+str(round(percentages[c],2))+"%"
                        print(string)
                        textsurface1 = myfont_percent.render(string, False, WHITE)
                        screen.blit(textsurface1, (20, 280+50*c))
                    pygame.display.flip()

    if take_pic==0:
        # draw frame
        screen.blit(img, (200,200))
        pygame.display.flip()
        # grab next frame
        img = webcam.get_image()

    clock.tick(REFRESH_RATE)
