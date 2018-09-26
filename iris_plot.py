import matplotlib.pyplot as plt
import numpy as np

def plot_scatter1(train_x, train_t, validation_x, validation_t, train_setosa, train_versicolor, train_virginica, val_setosa, val_versicolor, val_virginica, comp, x, y):
    plt.plot(train_x[train_setosa,x],     train_x[train_setosa,y],    'o',mfc='red',  mec='red')
    plt.plot(train_x[train_versicolor,x], train_x[train_versicolor,y],'v',mfc='blue', mec='blue')
    plt.plot(train_x[train_virginica,x],  train_x[train_virginica,y], '^',mfc='green',mec='green')
    """
    plt.plot(validation_x[np.logical_and(val_setosa,     comp[:,0]),x], validation_x[np.logical_and(val_setosa,     comp[:,0]),y],'o',mfc='white',mec='red')
    plt.plot(validation_x[np.logical_and(val_versicolor, comp[:,1]),x], validation_x[np.logical_and(val_versicolor, comp[:,1]),y],'v',mfc='white',mec='blue')
    plt.plot(validation_x[np.logical_and(val_virginica,  comp[:,2]),x], validation_x[np.logical_and(val_virginica,  comp[:,2]),y],'^',mfc='white',mec='green')
    """
    """
    plt.plot(validation_x[np.logical_and(val_setosa,     np.logical_not(comp[:,0])),x], validation_x[np.logical_and(val_setosa,     np.logical_not(comp[:,0])),y],'o',mfc='black',mec='red')
    plt.plot(validation_x[np.logical_and(val_versicolor, np.logical_not(comp[:,1])),x], validation_x[np.logical_and(val_versicolor, np.logical_not(comp[:,1])),y],'v',mfc='black',mec='blue')
    plt.plot(validation_x[np.logical_and(val_virginica,  np.logical_not(comp[:,2])),x], validation_x[np.logical_and(val_virginica,  np.logical_not(comp[:,2])),y],'^',mfc='black',mec='green')
    """
def plot_scatter2(train_x, train_t, validation_x, validation_t, train_setosa, train_versicolor, train_virginica, val_setosa, val_versicolor, val_virginica, comp, x, y):
    plt.plot(train_x[train_setosa,x],     train_x[train_setosa,y],    'o',mfc='red',  mec='red')
    plt.plot(train_x[train_versicolor,x], train_x[train_versicolor,y],'v',mfc='blue', mec='blue')
    plt.plot(train_x[train_virginica,x],  train_x[train_virginica,y], '^',mfc='green',mec='green')

    plt.plot(validation_x[np.logical_and(val_setosa,     comp[:,0]),x], validation_x[np.logical_and(val_setosa,     comp[:,0]),y],'o',mfc='white',mec='red')
    plt.plot(validation_x[np.logical_and(val_versicolor, comp[:,1]),x], validation_x[np.logical_and(val_versicolor, comp[:,1]),y],'v',mfc='white',mec='blue')
    plt.plot(validation_x[np.logical_and(val_virginica,  comp[:,2]),x], validation_x[np.logical_and(val_virginica,  comp[:,2]),y],'^',mfc='white',mec='green')


    plt.plot(validation_x[np.logical_and(val_setosa,     np.logical_not(comp[:,0])),x], validation_x[np.logical_and(val_setosa,     np.logical_not(comp[:,0])),y],'o',mfc='black',mec='red')
    plt.plot(validation_x[np.logical_and(val_versicolor, np.logical_not(comp[:,1])),x], validation_x[np.logical_and(val_versicolor, np.logical_not(comp[:,1])),y],'v',mfc='black',mec='blue')
    plt.plot(validation_x[np.logical_and(val_virginica,  np.logical_not(comp[:,2])),x], validation_x[np.logical_and(val_virginica,  np.logical_not(comp[:,2])),y],'^',mfc='black',mec='green')


def plot_scatter3(train_x, train_t, validation_x, validation_t, train_setosa, train_versicolor, train_virginica, val_setosa, val_versicolor, val_virginica, comp, x, y):
    """
    plt.plot(train_x[train_setosa,x],     train_x[train_setosa,y],    'o',mfc='red',  mec='red')
    plt.plot(train_x[train_versicolor,x], train_x[train_versicolor,y],'v',mfc='blue', mec='blue')
    plt.plot(train_x[train_virginica,x],  train_x[train_virginica,y], '^',mfc='green',mec='green')

    plt.plot(validation_x[np.logical_and(val_setosa,     comp[:,0]),x], validation_x[np.logical_and(val_setosa,     comp[:,0]),y],'o',mfc='white',mec='red')
    plt.plot(validation_x[np.logical_and(val_versicolor, comp[:,1]),x], validation_x[np.logical_and(val_versicolor, comp[:,1]),y],'v',mfc='white',mec='blue')
    plt.plot(validation_x[np.logical_and(val_virginica,  comp[:,2]),x], validation_x[np.logical_and(val_virginica,  comp[:,2]),y],'^',mfc='white',mec='green')
    """
    plt.plot(validation_x[np.logical_and(val_setosa,     np.logical_not(comp[:,0])),x], validation_x[np.logical_and(val_setosa,     np.logical_not(comp[:,0])),y],'o',mfc='black',mec='red')
    plt.plot(validation_x[np.logical_and(val_versicolor, np.logical_not(comp[:,1])),x], validation_x[np.logical_and(val_versicolor, np.logical_not(comp[:,1])),y],'v',mfc='black',mec='blue')
    plt.plot(validation_x[np.logical_and(val_virginica,  np.logical_not(comp[:,2])),x], validation_x[np.logical_and(val_virginica,  np.logical_not(comp[:,2])),y],'^',mfc='black',mec='green')

def plot_eval(model, train_x, train_t,  validation_x, validation_t):
    plt.figure(1)
    # ['setosa', 'versicolor', 'virginica'],
    train_setosa = train_t[:,0]==1
    train_versicolor = train_t[:,1]==1
    train_virginica = train_t[:,2]==1

    val_setosa = validation_t[:,0]==1
    val_versicolor = validation_t[:,1]==1
    val_virginica = validation_t[:,2]==1

    predict = model.predict(validation_x)
    comp = predict==validation_t

    # sepal length vs sepal width
    plt.subplot(3,3,1)
    plot_scatter2(train_x, train_t, validation_x, validation_t, train_setosa, train_versicolor, train_virginica, val_setosa, val_versicolor, val_virginica, comp, 0, 1)

    # sepal length vs petal length
    plt.subplot(3,3,4)
    plot_scatter2(train_x, train_t, validation_x, validation_t, train_setosa, train_versicolor, train_virginica, val_setosa, val_versicolor, val_virginica, comp, 0, 2)

    # sepal length vs petal width
    plt.subplot(3,3,7)
    plot_scatter2(train_x, train_t, validation_x, validation_t, train_setosa, train_versicolor, train_virginica, val_setosa, val_versicolor, val_virginica, comp, 0, 3)

    #  sepal width vs petal length
    plt.subplot(3,3,5)
    plot_scatter2(train_x, train_t, validation_x, validation_t, train_setosa, train_versicolor, train_virginica, val_setosa, val_versicolor, val_virginica, comp, 1, 2)

    #  sepal width vs petal width
    plt.subplot(3,3,8)
    plot_scatter2(train_x, train_t, validation_x, validation_t, train_setosa, train_versicolor, train_virginica, val_setosa, val_versicolor, val_virginica, comp, 1, 3)

    #  petal length vs petal width
    plt.subplot(3,3,9)
    plot_scatter2(train_x, train_t, validation_x, validation_t, train_setosa, train_versicolor, train_virginica, val_setosa, val_versicolor, val_virginica, comp, 2, 3)
