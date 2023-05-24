# -*- coding: utf-8 -*-
"""

@author: 3laa
"""
import tkinter as tk
from tkinter import filedialog, messagebox ,ttk
import seaborn as sns
import pandas as pd
import ipywidgets as widgets
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE



df = pd
root=tk.Tk()
root = tk.Tk()
root.geometry("500x500") 
root.pack_propagate(False) 
root.configure(bg="lightgray")
root.resizable(0, 0) 
################################################ main page 
label_file = tk.Label(root, text="No File Selected",fg="navy",bg="lightgray")
label_file.place(rely=0, relx=0.35)
button1 = tk.Button(root, text="Browse A File",fg="white",bg="black", command=lambda: File_dialog())
button1.place(rely=0.07, relx=0.45)
button2 = tk.Button(root, text="Regression",fg="white",bg="black", command=lambda: Regression())
button2.place(rely=0.14, relx=0.45)
button3 = tk.Button(root, text="KNN",fg="white",bg="black", command=lambda: KNN())
button3.place(rely=0.21, relx=0.45)
button4 = tk.Button(root, text="preprocessing",fg="white",bg="black", command=lambda: preprocess())
button4.place(rely=0.28, relx=0.45)
button5 = tk.Button(root, text="show data",fg="white",bg="black", command=lambda: show_data())
button5.place(rely=0.35, relx=0.45)
button6 = tk.Button(root, text="draw data",fg="white",bg="black", command=lambda: draw_data())
button6.place(rely=0.42, relx=0.45)
button7 = tk.Button(root, text="PCA",fg="white",bg="black", command=lambda: pca())
button7.place(rely=0.49, relx=0.45)
button8 = tk.Button(root, text="ERROR",fg="white",bg="black", command=lambda: error())
button8.place(rely=0.56, relx=0.45)
button9 = tk.Button(root, text="over&&under",fg="white",bg="black", command=lambda: over())
button9.place(rely=0.63, relx=0.45)
button10 = tk.Button(root, text="tree",fg="white",bg="black", command=lambda: tree())
button10.place(rely=0.70, relx=0.45)
def File_dialog():
    global df
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select A File",
                                          filetype=(("csv files", "*.csv"),("All Files", "*.*")))
    label_file["text"] = filename
    file_path = label_file["text"]
    try:
        excel_filename = r"{}".format(file_path)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)

    except ValueError:
        tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No such file as {file_path}")
        return None
    print(df)
    return None
def pca():
    global df
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    x=StandardScaler().fit_transform(x)
    
    #PCA
    pca=PCA(n_components=2)
    principalcomponents=pca.fit_transform(x)
    principalDf =pd.DataFrame(data=principalcomponents,columns=('p1','p2'))
    df =pd.concat([principalDf,y],axis=1)
    print(df)
def over():
    global df
    pltt = tk.Tk()
    pltt.title("Scatter and Line Plot")
    pltt.geometry("500x250") 
    pltt.pack_propagate(False) 
    pltt.configure(bg="lightgray")
    pltt.resizable(0, 0)
    errors=tk.LabelFrame(pltt,text="ESRRORS")
    errors.place(height=150, width=400, rely=0.0, relx=0.0)
    label_file3=ttk.Label(errors)
    label_file3.place(rely=0,relx=0)
    le=LabelEncoder()
    df.iloc[:,-1]=le.fit_transform(df.iloc[:,-1])
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
    print("Before UnderSampling: ",Counter(y_train))
    undersample=RandomUnderSampler(sampling_strategy="majority")
    x_train_under,y_train_under=undersample.fit_resample(x_train, y_train)
    print("After UnderSampling: ",Counter(y_train_under))
    print("Before OverSampling: ",Counter(y_train))
    
    smote=SMOTE()
    #fit and apply the transform 
    
    x_train_over,y_train_over=smote.fit_resample(x_train, y_train)
    
    #Summarize class Distribution
    
    print("After OverSampling: ",Counter(y_train_over))
    label_file3["text"]="Before UnderSampling: "+str(Counter(y_train))+"\nAfter UnderSampling: "+str(Counter(y_train_under))+"\nBefore OverSampling: "+str(Counter(y_train))+"\nAfter OverSampling: "+str(Counter(y_train_over))

def show_data():
    global df
    table=tk.Tk()
    table = tk.Tk()
    table.geometry("500x500") 
    table.configure(bg="lightgray")
    table.pack_propagate(False) 
    table.resizable(0, 0) 
    frame1 = tk.LabelFrame(table, text="Excel Data")
    frame1.place(height=250, width=500)

    tv1 = ttk.Treeview(frame1)
    tv1.place(relheight=1, relwidth=1)

    treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) 
    treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) 
    tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) 
    treescrollx.pack(side="bottom", fill="x") 
    treescrolly.pack(side="right", fill="y") 
    
    tv1.delete(*tv1.get_children())
    tv1["column"] = list(df.columns)
    print(list(df.columns))
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)

    df_rows = df.to_numpy().tolist() 
    for row in df_rows:
        tv1.insert("", "end", values=row) 
    errors=tk.LabelFrame(table,text="Describtion")
    errors.place(height=250, width=400, rely=0.5, relx=0)
    label_file3=ttk.Label(errors)
    label_file3.place(rely=0,relx=0)
    ifnullex=df.isnull().any()
    # Count the occurrences of each class label
    class_counts = df[column].value_counts()
    # Determine if the dataset is balanced or not
    is_balanced = class_counts.nunique() == 1
    label_file3["text"]="Number of columns : "+str(len(df.columns))+"\nNumber of rows : "+str(len(df))+"\nSize :"+str(df.shape)+"\nType :"+str(type(df))+"\nMissing Value :"+str(ifnullex)+"\n Balanced :"+str(is_balanced)+"\nDistinct Values for Each Attribute: \n"+str(df.nunique())                          
    return None

def error(): 
    sle=tk.Tk()
    sle.title("Select Error")
    sle.geometry("250x250") 
    sle.pack_propagate(False) 
    sle.configure(bg="lightgray")
    sle.resizable(0, 0) 
    radio_var = tk.StringVar()
    radio1 = tk.Radiobutton(sle, text="MSE", variable=radio_var, value="MSE",bg="lightgray")
    radio1.select()
    radio2 = tk.Radiobutton(sle, text="RMSE", variable=radio_var, value="RMSE",bg="lightgray")
    radio3 = tk.Radiobutton(sle, text="MAE", variable=radio_var, value="MAE",bg="lightgray")
    radio1.pack()
    radio2.pack()
    radio3.pack()
    selected_value = radio_var.get()
    button1 = tk.Button(sle, text="show result",fg="white",bg="black", command=lambda: showerror(selected_value))
    button1.place(rely=0.35, relx=0.35)
def showerror(selected_value):
    global df
    pltt = tk.Tk()
    pltt.title("Scatter and Line Plot")
    pltt.geometry("250x250") 
    pltt.pack_propagate(False) 
    pltt.configure(bg="lightgray")
    pltt.resizable(0, 0) 
    errors=tk.LabelFrame(pltt,text="ESRRORS")
    errors.place(height=150, width=400, rely=0.0, relx=0.0)
    label_file3=ttk.Label(errors,text="train error : \ntest error : ")
    label_file3.place(rely=0,relx=0)

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    regr=SGDRegressor(learning_rate='constant', eta0=1, max_iter=1000)
    regr.fit(X_train,y_train)
    y_test_pred  = regr.predict(X_test)
    y_train_pred =regr.predict(X_train)
    
    if selected_value=="MSE":
        test_mse=mean_squared_error(y_test,y_test_pred)
        train_mse=mean_squared_error(y_train,y_train_pred)
        label_file3["text"]="train error : "+str(test_mse)+"\ntest error : "+str(train_mse)
    elif selected_value=="RMSE":
        test_rmse=math.sqrt(mean_squared_error(y_test,y_test_pred))
        train_rmse=math.sqrt(mean_squared_error(y_train,y_train_pred))
        label_file3["text"]="train error : "+str(test_rmse)+"\ntest error : "+str(train_rmse)
    else:
        test_mae=mean_absolute_error(y_test,y_test_pred)
        train_mae=mean_absolute_error(y_train,y_train_pred)
        label_file3["text"]="train error : "+str(test_mae)+"\ntest error : "+str(train_mae)
  
def Regression():
    global df
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    regr=LinearRegression()
    regr.fit(X_train,y_train)
    #pred=regr.predict(np.array([[1.4]]))
    y_pred =regr.predict(X_test)
    pltt = tk.Tk()
    pltt.title("Scatter and Line Plot")
    pltt.geometry("500x500") 
    pltt.pack_propagate(False) 
    pltt.configure(bg="lightgray")
    pltt.resizable(0, 0) 
    # Create a figure and axes for the plots
    fig = plt.Figure( dpi=100)
    ax_line = fig.add_subplot(121.2)
    
    # Create the scatter plot
    ax_line.scatter(X_test,y_test , c='blue')
    ax_line.set_xlabel('X-axis')
    ax_line.set_ylabel('Y-axis')
   
    # Create the line plot
    ax_line.plot(X_test,y_pred, c='black')
    ax_line.set_title('Plot')
    
    # Create a canvas to display the plots in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=pltt)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    
def KNN():
    cla=tk.Tk()
    cla.title("KNN")
    cla.geometry("400x400") 
    cla.configure(bg="lightgray")
    cla.pack_propagate(False) 
    cla.resizable(0, 0) 
    la = tk.Label(cla,fg="navy",bg="lightgray")
    la.place(rely=0, relx=0.1)
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
    
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(x, y)
    predictions=model.predict(X_test)
    matrix=confusion_matrix( y_test, predictions)
    print(matrix)
    acc=accuracy_score(y_test, predictions)
    pre=precision_score( y_test, predictions)
    rec=recall_score( y_test, predictions)
    f1=f1_score( y_test, predictions)
    print(acc)
    print(pre)
    print(rec)
    print(f1)
    from sklearn.metrics import classification_report
    report = classification_report(y_test, predictions)
    la["text"]="Confusion_matrix is   "+str(matrix)+"\n Accuracy_score is "+str(acc)+"\n precision_score is "+str(pre)+"\n recall_score is "+str(rec)+"\n f1_score is "+str(f1)+"\n REPORT :"+str(report)
def preprocess():
    global df,entry
    p=tk.Tk()
    p.title("Preprocessing")
    p.geometry("250x350") 
    p.pack_propagate(False) 
    p.configure(bg="lightgray")
    p.resizable(0, 0) 
    lbl = tk.Label(p, text="Enter Column Number",fg="navy",bg="lightgray")
    lbl.place(rely=0.1, relx=0.05)
    entry = tk.Entry(p)
    entry.place(height=20,width=75,rely=0.1,relx=0.55)
    ty=tk.LabelFrame(p)
    ty.place(height=150, width=150, rely=0.2, relx=0.05)
    radio_var = tk.StringVar()
    radio1 = tk.Radiobutton(ty, text="SimpleImputer", variable=radio_var, bg="lightgray",value="sim")
    radio1.select()
    radio2 = tk.Radiobutton(ty, text="OneHotEncoder", variable=radio_var, bg="lightgray",value="ohe")
    radio3 = tk.Radiobutton(ty, text="LabelEncoder", variable=radio_var, bg="lightgray",value="le")
    radio4 = tk.Radiobutton(ty, text="StandardScaler", variable=radio_var, bg="lightgray",value="N")
    radio5 = tk.Radiobutton(ty, text="MinMax", variable=radio_var, bg="lightgray",value="M")
    radio1.pack()
    radio2.pack()
    radio3.pack()
    radio4.pack()
    radio5.pack()
    button1 = tk.Button(p, text="Preprocess",fg="white",bg="black", command=lambda: prepp(radio_var.get(),int(entry.get())))
    button1.place(rely=0.65, relx=0.05)
    button2 = tk.Button(p, text="Show Data",fg="white",bg="black", command=lambda: show_data())
    button2.place(rely=0.85, relx=0.05)
def prepp(selected,num):
    global df
    if selected=="sim":
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(df.iloc[:,num:num+1 ])
        df.iloc[:, num:num+1] = imputer.transform(df.iloc[:,num:num+1])
        print(df)
    elif selected=="ohe":
        ct = ColumnTransformer([('encoder', OneHotEncoder(), [num])], remainder='passthrough')
        df = pd.DataFrame(ct.fit_transform(df))
        print(df)
    elif selected=="le":
        le = LabelEncoder()
        df.iloc[:,num] = le.fit_transform(df.iloc[:,num])
        print(df)
    elif selected=="N":
        st= StandardScaler(copy=True,with_mean=True,with_std=True) 
        df.iloc[:, :-1]=pd.DataFrame(st.fit_transform(df.iloc[:, :-1].values))
        print(df)
        Data_normalizer = Normalizer(norm='l1').fit(df.iloc[:, :-1].values)
        df.iloc[:, :-1] = pd.DataFrame(Data_normalizer.transform(df.iloc[:, :-1].values))
        print(df)
    else:
        scaler = MinMaxScaler()
        df= pd.DataFrame(scaler.fit_transform(df)) 
        print(df)
def draw_data():
    p=tk.Tk()
    p.title("Preprocessing")
    p.geometry("250x350") 
    p.pack_propagate(False) 
    p.configure(bg="lightgray")
    p.resizable(0, 0)
    but1 = tk.Button(p, text="Line plot",fg="white",bg="black", command=lambda: lin())
    but1.place(rely=0.10, relx=0.45)
    but2 = tk.Button(p, text="Box plot",fg="white",bg="black", command=lambda: bx())
    but2.place(rely=0.25, relx=0.45)
    but3 = tk.Button(p, text="Scatter plot",fg="white",bg="black", command=lambda: scat())
    but3.place(rely=0.40, relx=0.45)
    but4 = tk.Button(p, text="Histogram",fg="white",bg="black", command=lambda: ht())
    but4.place(rely=0.55, relx=0.45)
    but4 = tk.Button(p, text="Correlation Heatmap",fg="white",bg="black", command=lambda: cor())
    but4.place(rely=0.70, relx=0.45)
def lin():
    pp=tk.Tk()
    pp.title("Line Plot")
    pp.geometry("250x250") 
    pp.pack_propagate(False) 
    pp.configure(bg="lightgray")
    pp.resizable(0, 0)
    label_file = tk.Label(pp, text="X",fg="navy",bg="lightgray")
    label_file.place(rely=0.1, relx=0)
    label_file = tk.Label(pp, text="Y",fg="navy",bg="lightgray")
    label_file.place(rely=0.2, relx=0)
    label_file = tk.Label(pp, text="Target",fg="navy",bg="lightgray")
    label_file.place(rely=0.3, relx=0) 
    entry = tk.Entry(pp)
    entry.place(height=20,width=75,rely=0.1,relx=0.1)
    entry2 = tk.Entry(pp)
    entry2.place(height=20,width=75,rely=0.2,relx=0.1)
    entry3 = tk.Entry(pp)
    entry3.place(height=20,width=75,rely=0.3,relx=0.15)
    but1 = tk.Button(pp, text="Draw Line Plot",fg="white",bg="black", command=lambda: line(int(entry.get()),int(entry2.get()),int(entry3.get())))
    but1.place(rely=0.45, relx=0.1)   
def cor():
    fig, ax = plt.subplots()
    #sns.lineplot(df[df.columns[x]],df[df.columns[y]],hue=df[df.columns[z]], ax=ax)
    correlation_matrix = df.corr()

# Create a correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',ax=ax)
    
    ax.set_title('Correlation Heatmap')
    
    # Create a Tkinter window
    window = tk.Tk()
    window.title('Correlation Heatmap')
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    # Run the Tkinter event loop
    tk.mainloop()       
def line(x,y,z):
    global df
    # Create a seaborn line plot
    fig, ax = plt.subplots()
    sns.lineplot(df[df.columns[x]],df[df.columns[y]],hue=df[df.columns[z]], ax=ax)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Line Plot')
    
    # Create a Tkinter window
    window = tk.Tk()
    window.title('Line Plot')
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    # Run the Tkinter event loop
    tk.mainloop()  
def bx():
    pp=tk.Tk()
    pp.title("Box plot")
    pp.geometry("150x150") 
    pp.pack_propagate(False) 
    pp.configure(bg="lightgray")
    pp.resizable(0, 0)
    label_file = tk.Label(pp, text="Coulmn",fg="navy",bg="lightgray")
    label_file.place(rely=0.1, relx=0) 
    label_file = tk.Label(pp, text="Target",fg="navy",bg="lightgray")
    label_file.place(rely=0.3, relx=0)
    entry = tk.Entry(pp)
    entry2 = tk.Entry(pp)
    entry2.place(height=20,width=75,rely=0.3,relx=0.3)
    entry.place(height=20,width=75,rely=0.1,relx=0.3)
    but1 = tk.Button(pp, text="Draw Box Plot",fg="white",bg="black", command=lambda: box(int(entry.get()),int(entry2.get())))
    but1.place(rely=0.5, relx=0.35)          
def box(x,z):
    global df
    # Create a seaborn line plot
    fig, ax = plt.subplots()
    sns.boxplot(df[df.columns[x]],hue=df[df.columns[z]], ax=ax)
    ax.set_xlabel(df.columns[x])
    ax.set_title('Box Plot')
    
    # Create a Tkinter window
    window = tk.Tk()
    window.title('BOX Plot')
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    # Run the Tkinter event loop
    tk.mainloop()   
def scat():
    pp=tk.Tk()
    pp.title("Scatter Plot")
    pp.geometry("250x250") 
    pp.pack_propagate(False) 
    pp.configure(bg="lightgray")
    pp.resizable(0, 0)
    label_file = tk.Label(pp, text="X",fg="navy",bg="lightgray")
    label_file.place(rely=0.1, relx=0)
    label_file = tk.Label(pp, text="Y",fg="navy",bg="lightgray")
    label_file.place(rely=0.2, relx=0)
    label_file = tk.Label(pp, text="Target",fg="navy",bg="lightgray")
    label_file.place(rely=0.3, relx=0) 
    entry = tk.Entry(pp)
    entry.place(height=20,width=75,rely=0.1,relx=0.1)
    entry2 = tk.Entry(pp)
    entry2.place(height=20,width=75,rely=0.2,relx=0.1)
    entry3 = tk.Entry(pp)
    entry3.place(height=20,width=75,rely=0.3,relx=0.15)
    but1 = tk.Button(pp, text="Draw Scatter Plot",fg="white",bg="black", command=lambda: scatt(int(entry.get()),int(entry2.get()),int(entry3.get())))
    but1.place(rely=0.45, relx=0.1)        
def scatt(x,y,z):
    global df
    # Create a seaborn line plot
    fig, ax = plt.subplots()
    sns.scatterplot(df[df.columns[x]],df[df.columns[x]],hue=df[df.columns[z]], ax=ax)
    ax.set_xlabel('Scatter Plot')
    ax.set_title('Scatter Plot')
    # Create a Tkinter window
    window = tk.Tk()
    window.title('Scatter Plot')
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    # Run the Tkinter event loop
    tk.mainloop() 
def ht():
    pp=tk.Tk()
    pp.title("Histogram plot")
    pp.geometry("200x200") 
    pp.pack_propagate(False) 
    pp.configure(bg="lightgray")
    pp.resizable(0, 0)
    label_file = tk.Label(pp, text="Coulmn",fg="navy",bg="lightgray")
    label_file.place(rely=0.1, relx=0) 
    
    entry = tk.Entry(pp)
    entry.place(height=20,width=75,rely=0.1,relx=0.3)
    but1 = tk.Button(pp, text="Draw Histogram Plot",fg="white",bg="black", command=lambda: hst(int(entry.get())))
    but1.place(rely=0.5, relx=0.35)       
def hst(x):
    global df
    fig, ax = plt.subplots()
    plt.hist(df[df.columns[x]])
    ax.set_xlabel(df.columns[x])
    ax.set_title('Histogram Plot')
    
    # Create a Tkinter window
    window = tk.Tk()
    window.title('Histogram Plot')
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    # Run the Tkinter event loop
    tk.mainloop() 
def tree():
    global label_file
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    pltt = tk.Tk()
    pltt.title("Scatter and Line Plot")
    pltt.geometry("500x500") 
    pltt.pack_propagate(False) 
    pltt.configure(bg="lightgray")
    pltt.resizable(0, 0) 
    fig, ax = plt.subplots()
    col_names=['pregnant','ins','bmi','age','glucose','bp','perd','label']
    feature_cols=['pregnant','ins','bmi','age','glucose','bp','perd']
    df=pd.read_csv(label_file["text"],header=None,names=col_names)
    
    df.head();
    
    x=df[feature_cols]
    y=df.label
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
    
    
    clf=DecisionTreeClassifier()
    clf=clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    
    
    print("Accurancy:",metrics.accuracy_score(y_test,y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    
    
    plt.figure(figsize=(50,50))
    a=plot_tree(clf,feature_names=feature_cols,class_names=['0','1'],filled=True,rounded=True,fontsize=14,ax=ax)
    errors=tk.LabelFrame(pltt)
    errors.place(height=200, width=400, rely=0.6, relx=0.0)
    label_file3=ttk.Label(errors)
    label_file3.place(rely=0,relx=0)
    label_file3["text"]="Accurancy:"+str(metrics.accuracy_score(y_test,y_pred))+"\nConfusion Matrix:"+str(confusion_matrix(y_test, y_pred))+"\n"+str(metrics.classification_report(y_test, y_pred))
    canvas = FigureCanvasTkAgg(fig, master=pltt)
    canvas.draw()
    canvas.get_tk_widget().pack()
       
root.mainloop()