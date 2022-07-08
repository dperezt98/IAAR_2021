# DANIEL PÉREZ RODRÍGUEZ - PRÁCTICA IAAR 2021
import jetson.inference
import jetson.utils
from adafruit_servokit import ServoKit # Importamos la libreria para usar servos

kit = ServoKit(channels=16) # Especificamos el número de pines de la placa gpio, en nuestro caso 16
argv=['--model=/home/alumno1/Descargas/jetson-inference/python/training/detection/ssd/models/person/ssd-mobilenet.onnx','--labels=/home/alumno1/Descargas/jetson-inference/python/training/detection/ssd/models/person/labels.txt','--input-blob=input_0','--output-cvg=scores', '--output-bbox=boxes']
net = jetson.inference.detectNet("ssd-mobilenet-v2", argv, threshold=0.5)
#net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5) # Modelo a usar, theshold: porcentaje mínimo de confianza sobre un objeto

# Si no quisieramos parametros tendriamos que poner:videoSource("csi://0")
# Parametros de video source: videoSource("csi://0", argv=['--input-flip=rotate-180', '--input-width=1280', '--input-height=720', '--input-frameRate=30'])
camera = jetson.utils.videoSource("csi://0", argv=['--input-flip=rotate-180']) # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("rtp://192.168.55.100:1234") # stream output

width = 0 # Ancho de la imagen obtenida por la camara
height = 0  # Alto de la imagen obtenida por la camara
screenSize = 0 # Area de la pantalla en píxeles
pixelAngleRatio = 75 # Conversion pixel/angulo

maxPan = 179 # Máxima rotación del servo en el eje y
minPan = 1 # Mínima rotación del servo en el eje y
maxTilt = 179 # Máxima rotación del servo en el eje x
minTilt = 1 # Mínima rotación del servo en el eje x

defaultPan = 90 # Posición por defecto del pan
actualPan = defaultPan # Valor actual del pan
defaultTilt = 90 # Valor por defecto del tilt
actualTilt = defaultTilt # Valor actual del tilt
actualPanDirection = -1 # 1 significa que gira a la izquierda. -1 Que gira a la derecha

constantDeadZone = 35
maxErrorPan = constantDeadZone # pixel error. Error máximo posible respecto al centro de la persona reconocida en el eje X
maxErrorTilt = constantDeadZone # pixel error. Error máximo posible respecto al centro de la persona reconocida en el eje Y

numImageWithoutPerson = 0 # Contador de images tras perder a un objetivo(persona)
maxNumImageWithoutPerson = 10 # Número máximo de images máximo tras perder aun objetivo(persona)

personCode = 1  # Código querepresenta a una persona de todos los objetos detectados por ssd-mobilenet-v2(1) - myModel(1)
personZoneToFind = "top" # Puede ser "top" o "center"
thereIsSomeone = False # Variable de control. Si se encuentra a una persona en los objetos detectados estará a True

def findPeople(anglePanPosition, angleTiltPosition, controlPanDirection):
    """Devuelve la siguiente posición en la que debe estar la cámara en su ruta
    para encontrar a una persona. Este movimiento es horizontal, de derecha a izquierda. Una
    vez completado repite el giro en el sentido contrario. Devuelve (panPosition, tiltPosition, controlPanDirection)
    """
    if(controlPanDirection == 1):
        panPosition = anglePanPosition + 1
        if panPosition >= maxPan:
            controlPanDirection = -1
    else:
        panPosition = anglePanPosition - 1
        if panPosition < minPan:
            controlPanDirection = 1

    # Tilt debe volver a su posición por defecto
    if (angleTiltPosition < defaultTilt + 2) & (angleTiltPosition > defaultTilt - 2):
        tiltPosition = defaultTilt
    else:
        if angleTiltPosition > defaultTilt:
            tiltPosition = angleTiltPosition - 2
        else:
            tiltPosition = angleTiltPosition + 2


    return (panPosition, tiltPosition, controlPanDirection)

def thereIsPeopleOnDetections(detections):
    """Comprueba si se ha reconido a una persona en una lista de detections"""
    for i in detections:
        if i.ClassID==personCode:
            return True
    
    return False
        
def showClassID(detections):
    """Muestra la clase de todos los objetos reconocidos en la imagen"""
    for i in detections:
        print("Class detected: ", i.ClassID)

def selectPerson(detections):
    """Selecciona la persona más cercana a la cámara. Que en nuestro caso será la que tenga
    un área más grande (area de su bounding box)"""
    # De todos los objetos reconocidos debemos seleccionar a la persona más cercana a la cámara
    # El id de una persona es el valor 1
    maxArea = -1
    for i in detections:
        if i.ClassID == personCode:
            if maxArea < i.Area:
                personSelected = i
                maxArea = i.Area

    return personSelected

def getBboxTopZone(person):
    """Devuelve la posición de la parte central alta del bounding box"""
    centerX, centerY = person.Center
    centerY = centerY - person.Height/4

    return centerX, centerY

def deadZones(person):
    """Devuelve como de grande es el cuadrado/rectángulo en el que la cámara detecta que esta apuntando al centro del objetivo"""
    bboxSize = person.Width*person.Height
    ratioDeadZoneSize = bboxSize/screenSize
    maxEP = constantDeadZone + (constantDeadZone*ratioDeadZoneSize)
    maxET = constantDeadZone + (constantDeadZone*ratioDeadZoneSize)
    return (maxEP,maxET)

def moveCameraTo(centerX, centerY, actualPan, actualTilt):
    """Calcula la posición de la cámara dado el centroide del objeto a seguir"""
    errorPan = centerX - width/2
    errorTilt = centerY - height/2
    if abs(errorPan) > maxErrorPan:
        newPan = actualPan - errorPan/pixelAngleRatio
    else:
        newPan = actualPan
    
    if abs(errorTilt) > maxErrorTilt:
        newTilt = actualTilt + errorTilt/pixelAngleRatio
    else:
        newTilt = actualTilt

    return (newPan, newTilt)

def checkCameraAngle(pan, tilt):
    """Comprueba que los ángulos estén en el rango 0-180. Devuelve (pan,tilt)"""
    returnPan = pan
    returnTilt = tilt
    if returnPan >= maxPan:
        returnPan = maxPan
    if returnPan <= minPan:
        returnPan = minPan
    if returnTilt >= maxTilt:
        returnTilt = maxTilt
    if returnTilt <= minTilt:
        returnTilt = minTilt

    return (returnPan, returnTilt)

# Inicializamos las variables 
img = camera.Capture()
width = img.width
height = img.height
screenSize = width*height

while True:

    # Capturamos una imagen con la webcam
    img = camera.Capture()

    # Detectamos los objetos de dicha imagen
    detections = net.Detect(img)
    #showClassID(detections)

    # Si se ha recozido a alguna persona debemos trackearla. En caso contrario vigilaremos la zona
    if thereIsPeopleOnDetections(detections) is True:
        person = selectPerson(detections)
        # Estipulamos que parte de su cuerpo queremos buscar("top" o "center")
        if personZoneToFind == "top":
            centerX, centerY = getBboxTopZone(person)
        else:
            centerX, centerY = person.Center
        
        print("Person selected coordinates(x,y): (",centerX,",",centerY,") - Area: ", person.Area) 

        # Comprobamos como de grande es la zona a la que puede apuntar la camara     
        maxErrorPan, maxErrorTilt = deadZones(person)

        # Comprobamos la nueva posición que debe tener la cámara
        actualPan, actualTilt = moveCameraTo(centerX, centerY, actualPan, actualTilt)
        
        # Comprobamos que los ángulos calculados sean válidos
        actualPan, actualTilt = checkCameraAngle(actualPan, actualTilt)
        print("Pan:", actualPan, " - Tilt:", actualTilt)
        kit.servo[0].angle = actualPan
        kit.servo[1].angle = actualTilt

        # Resetamos la variable de imagenes sin detecciones
        numImageWithoutPerson = 0
    else:
        if numImageWithoutPerson >= maxNumImageWithoutPerson:  
            actualPan, actualTilt, actualPanDirection = findPeople(actualPan, actualTilt, actualPanDirection)
            # Comprobamos que los angulos calculados sean válidos
            actualPan, actualTilt = checkCameraAngle(actualPan, actualTilt)
            kit.servo[0].angle = actualPan
            kit.servo[1].angle = actualTilt
        else:
            numImageWithoutPerson = numImageWithoutPerson + 1
            print("Image after lose the track object:", numImageWithoutPerson)
            if numImageWithoutPerson >= maxNumImageWithoutPerson:
                print("The track object is missing...")
                print("Finding people...")
    
    # Mostramos la imagen por pantalla
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

    # Muesta la información de rendimiento de la red
    # net.PrintProfilerTimes()

    # Termina el programa con input/output EOS
    if not camera.IsStreaming() or not display.IsStreaming():
       break

