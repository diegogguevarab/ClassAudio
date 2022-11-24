# Librerías utilizadas 
from tkinter.font import BOLD, Font
from PIL import ImageTk, Image
from tkinter import filedialog
import tkinter.messagebox
from tkinter.ttk import *
import tkinter as tk
from tkinter import ttk
from tkinter import *
import classAudio
import contextlib
import datetime
import wave
import time

# Función encargada de arreglar las cadenas con los géneros resultantes
def styleGenreResult(entry_genres):
    final_genres = []
    for st in entry_genres:
        new_str = st.replace('[', '').replace(']', '').replace('\'', '')
        final_genres.append(new_str)
    return final_genres

# Función que verifica la duración del audio cargado
def checkAudioDuration(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

# Mensaje mientras procesa el archivo
def loadingMessage(event=None):
    return 'Analizando....'

# Alerta si no se escoge ningún archivo
def fileNotChoosed():
    tkinter.messagebox.showinfo("Error","Debe escoger un archivo para el análisis")

# Alerta si el archivo es de más de 30 segundos
def greaterThan30():
    tkinter.messagebox.showinfo("Error","El audio debe ser de máximo 30 segundos")


# Función principal que se encarga del manejo de la interfaz con tkinter
def interface():

    # Función para cerrar la ventana
    def Close():
        root.destroy()

    # Función para cargar los gráficos generados en el análisis del audio
    def write_slogan():
        nwin = Toplevel()
        nwin.title("Información del audio")
        imageLabel = tk.Label(nwin)
        imageLabel.pack()  
        image = tk.PhotoImage(file = "./Graphs.png")
        imageLabel.configure(image = image)
        imageLabel.image = image

    # Función para cargar las matrices de confusión generadas
    def see_metrics():
        nwin1 = Toplevel()
        nwin1.title("Matrices de Confusión de los modelos")
        imageLabel1 = tk.Label(nwin1)
        imageLabel1.pack()  
        image1 = ImageTk.PhotoImage(Image.open("./confusion.png"))
        imageLabel1.configure(image = image1)
        imageLabel1.image = image1

    # Barra de progreso
    def progressBar():
        pb1 = Progressbar(
            root, 
            orient=HORIZONTAL, 
            length=300, 
            mode='determinate'
        )
        pb1.place(x=210, y=260, width=200)
        for i in range(5):
            root.update_idletasks()
            pb1['value'] += 20
            time.sleep(1)
        pb1.destroy()
        loading.destroy()
        Label(root, text='Audio analizado con éxito!', foreground='green').place(x=225, y=200)

    # Función donde se sitúan los elementos del resultado de la predicción
    def placeElements(results_genre):
        results_title.config(text='Resultados', font=bold21)
        svm_title.config(text='SVM', font=bold15)
        rf_title.config(text='Random Forest', font=bold15)
        rn_title.config(text='MLP', font=bold15)
        knn_title.config(text='KNN', font=bold15)
        all_title.config(text='All', font=bold15)
        svm_genre.config(text=results_genre[0],font=13)
        rf_genre.config(text=results_genre[1], font=13)
        rn_genre.config(text=results_genre[2], font=13)
        knn_genre.config(text=results_genre[3], font=13)
        all_genre.config(text=results_genre[4], font=13)
        svm_acc.config(text=results_genre[5],font=13)
        rf_acc.config(text=results_genre[6], font=13)
        rn_acc.config(text=results_genre[7], font=13)
        knn_acc.config(text=results_genre[8], font=13)
        all_acc.config(text=results_genre[9], font=13)

        # Botón de ver gráficos
        plots = tk.Button(root)
        plots.place(x=100, y=450)
        plots.config(text='Ver gráficos del audio', command=write_slogan)

        # Botón de ver métricas
        metrics = tk.Button(root)
        metrics.place(x=350, y=450)
        metrics.config(text='Ver métricas', command=see_metrics)

        # Botón de salir
        exit = tk.Button(root)
        exit.place(x=270, y=500)
        exit.config(text='Salir', command=Close)


    # Función que muestra mensaje de analizando... o una alerta si el usuario no escogió el archivo
    # En este punto es llamada la función para predecir el género de la canción
    def press():
        if selected_file['text']:
            if checkAudioDuration(selected_file['text']) < 31:
                loading.config(text="Analizando....")
                progressBar()
                results_genre = classAudio.predictions(selected_file['text'])
                classAudio.descriptionGraph(selected_file['text'])
                placeElements(styleGenreResult(results_genre))
            else:
                greaterThan30()
        else:
            fileNotChoosed()

    # Abre el diálogo para la subida de archivos
    def UploadAction(event=None):
        filename = filedialog.askopenfilename(filetypes=[("Waveform audio", "*.wav")])
        if filename is not None:
            selected_file['text'] = filename

    # Uso de tkinter - Creación de ventana principal
    root = tk.Tk()
    root.title('Proyecto Final Aprendizaje de Máquina')
    root.geometry("600x600")

    # Titulo de la aplicacion
    bold21 = Font(size=21, weight=BOLD)
    window_title = tk.Label(text="Analizador de audio", font=bold21)
    window_title.place(x=210, y=20)
    
    # Reglas de la aplicacion
    instructions = tk.Label(text="Cargue un audio de máximo 30 segundos con extensión .wav:", font=('Helvetica',16))
    instructions.place(x=20, y=70)

    # Botón de subir archivo
    upload_file = tk.Button(root, text='Subir archivo (.wav)', command=UploadAction)
    upload_file.place(x=20, y=120)

    # Archivo seleccionado
    selected_file = tk.Label()
    selected_file.place(x=200, y=122)

    # Labels y estilo
    bold15 = Font(size=15, weight=BOLD)
    loading = Label(root)

    svm_title = Label(root)
    svm_genre = Label(root)
    rf_title = Label(root)
    rf_genre = Label(root)
    rn_title = Label(root)
    rn_genre = Label(root)
    knn_title = Label(root)
    knn_genre = Label(root)
    all_title = Label(root)
    all_genre = Label(root)
    svm_acc = Label(root)
    rf_acc = Label(root)
    rn_acc = Label(root)
    knn_acc = Label(root)
    all_acc = Label(root)

    results_title = Label(font=bold21)

    x_measure = 60
    y_measure = 320
    loading.place(x=270, y=200)
    results_title.place(x=250, y=270)

    # Modelos
    svm_title.place(x=x_measure, y=y_measure)
    rf_title.place(x=x_measure+80, y=y_measure)
    rn_title.place(x=x_measure+230, y=y_measure)
    knn_title.place(x=x_measure+330, y=y_measure)
    all_title.place(x=x_measure+430, y=y_measure)

    # Géneros predichos
    svm_genre.place(x=x_measure-10, y=y_measure+40)
    rf_genre.place(x=x_measure+110, y=y_measure+40)
    rn_genre.place(x=x_measure+220, y=y_measure+40)
    knn_genre.place(x=x_measure+320, y=y_measure+40)
    all_genre.place(x=x_measure+410, y=y_measure+40)

    # Accuracy de cada modelo
    svm_acc.place(x=x_measure-10, y=y_measure+70)
    rf_acc.place(x=x_measure+110, y=y_measure+70)
    rn_acc.place(x=x_measure+220, y=y_measure+70)
    knn_acc.place(x=x_measure+320, y=y_measure+70)
    all_acc.place(x=x_measure+410, y=y_measure+70)

    # Botón de iniciar análisis
    start_analysis = tk.Button(root, text='Empezar análisis', bg='red',command=press)
    start_analysis.place(x=240, y=160) 
        
    root.mainloop()

if __name__ == "__main__":
    print(datetime.datetime.now(), "Iniciando el programa...")
    interface()