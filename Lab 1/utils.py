from IPython import display
from matplotlib import pyplot as plt
import numpy as np
from scipy import io as sio


def loadDataset(datasetNr):
    """Loads specific dataset.

    Samples are in the 1st dimension (rows), and features in the
    2nd dimension. This convention must be consistent throughout the
    assignment; otherwise the plot code will break.

    Args:
        datasetNr (int [1-4]): Dataset to load.

    Returns:
        X (array): Data samples.
        D (array): Neural network target values.
        L (array): Class labels.
    """

    if not (1 <= datasetNr and datasetNr <= 4):
        raise ValueError("Unknown dataset number")

    data = sio.loadmat("Data/lab_data.mat")
    X = data[f"X{datasetNr}"]
    D = data[f"D{datasetNr}"]
    L = data[f"L{datasetNr}"].squeeze()

    return X.astype(float), D.astype(float), L.astype(int)


def plotDatasets():
    """Plots the datasets used in the assignment."""

    plotStrings = ["r.", "g.", "b."]
    c = "xo+*sd"

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    subplots = fig.subplots(2, 2)

    for (d, ax) in enumerate(subplots[:, :].flat):

        [X, _, L] = loadDataset(d + 1)

        # Plot first three datasets
        if d + 1 <= 3:
            ax.invert_yaxis()
            ax.set_title(f"Dataset {d+1}")
            for label in range(3):
                ind = (L == label).squeeze()
                ax.plot(X[ind,0], X[ind,1], plotStrings[label])
                # ax.plot(X[ind, 0], X[ind, 1], c[i])
                # ax.plot(X[ind, 0], X[ind, 1], ".")
        # Plot fourth dataset
        else:
            gridspec = ax.get_subplotspec().get_gridspec()
            ax.remove()
            subfig = fig.add_subfigure(gridspec[1, 1])
            subplots2 = subfig.subplots(4, 4)
            for (i, ax2) in enumerate(subplots2[:, :].flat):
                ax2.imshow(X[i].reshape(8, 8), cmap="gray")
                ax2.set_title(f"Class {L[i]}")
                ax2.set_axis_off()
            subfig.suptitle("Dataset 4")


def splitData(X, D, L, testFraction, seed=None):
    """Splits data into training and test portions.

    Args:
        X (array): Data samples.
        D (array): Neural network target values.
        L (array): Class labels.
        testFraction (float [0-1]): Fraction of data used for testing.
        seed (int): Used to enable reliable tests.

    Returns:
        XTrain (array): Training portion of X.
        DTrain (array): Training portion of D.
        LTrain (array): Training portion of L.
        XTest (array): Test portion of X.
        DTest (array): Test portion of D.
        LTest (array): Test portion of L.
    """

    nSamples = X.shape[0]

    if seed is None:
        perm = np.random.permutation(nSamples)
    else:
        perm = np.random.RandomState(seed=seed).permutation(nSamples)

    iTrain = sorted(perm[int(testFraction * nSamples) :])
    iTest = sorted(perm[: int(testFraction * nSamples)])

    return X[iTrain], D[iTrain], L[iTrain], X[iTest], D[iTest], L[iTest]


def splitDataEqualBins(X, D, L, nBins):
    """Splits data into separate equal-sized bins.

    Args:
        X (array): Data samples.
        D (array): Training targets for X.
        L (array): Data lables for X.
        nBins (int): Number of bins to split into.

    Returns:
        XBins (list): Output bins with data from X.
        DBins (list): Output bins with data from D.
        LBins (list): Output bins with data from L.
    """

    labels, counts = np.unique(L, return_counts=True)
    nLabels = labels.shape[0]

    nSamplesPerLabelPerBin = counts.min() // nBins

    # Get class labels
    labelInds = {}
    for label in labels:
        labelInds[label] = np.flatnonzero(L == label)
        np.random.shuffle(labelInds[label])

    XBins, DBins, LBins = [], [], []
    for m in range(nBins):
        sampleInds = np.concatenate(
            [a[m * nSamplesPerLabelPerBin : (m + 1) * nSamplesPerLabelPerBin] for a in labelInds.values()], axis=0,
        )

        XBins.append(X[sampleInds])
        DBins.append(D[sampleInds])
        LBins.append(L[sampleInds])

    return XBins, DBins, LBins


def splitDataBins(X, D, L, nBins):
    """Splits data into separate equal-sized bins.

    Args:
        X (array): Data samples.
        D (array): Training targets for X.
        L (array): Data lables for X.
        nBins: Number of bins to split into.

    Returns:
        XBins (list): Output bins with data from X.
        DBins (list): Output bins with data from D.
        LBins (list): Output bins with data from L.
    """

    nSamplesPerBin = X.shape[0] // nBins

    I = np.random.permutation(X.shape[0])

    XBins, DBins, LBins = [], [], []
    for b in range(nBins):
        sampleInds = I[b * nSamplesPerBin : (b + 1) * nSamplesPerBin]

        if X is not None:
            XBins.append(X[sampleInds])
        if D is not None:
            DBins.append(D[sampleInds])
        if L is not None:
            LBins.append(L[sampleInds])
    
    return XBins, DBins, LBins


def getCVSplit(XBins,DBins,LBins,nBins,i):
    """Combine data bins into training and validation sets
    for cross validation.

    Args:
        XBins (list of arrays): Binned data samples.
        DBins (list of arrays): Binned training targets for X.
        LBins (list of arrays): Binned lables for X.
        nBins (int): Number of bins in X, D, and L.
        i (int): Current cross-validation iteration.

    Returns:
        XTrain (array): Cross validation training data.
        DTrain (array): Cross validation training targets.
        LTrain (array): Cross validation training labels.
        XVal (array): Cross validation validation data.
        DVal (array): Cross validation validation targets.
        LVal (array): Cross validation validation labels.
    """
    
    if XBins is None:
        XTrain = None
        XVal = None
    else:
        XTrain = np.concatenate([XBins[j] for j in np.arange(nBins) if j != i])
        XVal = XBins[i]
        
    if DBins is None:
        DTrain = None
        DVal = None
    else:
        DTrain = np.concatenate([DBins[j] for j in np.arange(nBins) if j != i])
        DVal = DBins[i]
        
    if LBins is None:
        LTrain = None
        LVal = None
    else:
        LTrain = np.concatenate([LBins[j] for j in np.arange(nBins) if j != i])
        LVal = LBins[i]
    
    return XTrain, DTrain, LTrain, XVal, DVal, LVal


def plotResultsCV(meanAccs, kBest):
    """Plot accuracies and optimal k from the cross validation.
    """
    kBestAcc = meanAccs[kBest-1]
    kMax = np.size(meanAccs)
    
    plt.figure()
    plt.plot(np.arange(1, kMax+1), meanAccs, "k.-", label="Avg. val. accuracy")
    plt.plot(kBest, kBestAcc, 'bo', label=f"Max avg. val. accuracy, k={kBest}")
    plt.grid()
    plt.legend()
    plt.title(f'Maximum average cross-validation accuracy: {kBestAcc:.4f} for k = {kBest}')
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.show()

    
def _plotData(X, L, LPred):
    """Plot dataset 1, 2, or 3. Indicates correct and incorrect label predictions
    as green and red respectively.
    """

    c = "xo+*sd"

    for label in range(3):
        correctInd = (L == label) & (L == LPred)
        errorInd = (L == label) & (L != LPred)
        plt.plot(X[correctInd, 0], X[correctInd, 1], "g" + c[label])
        plt.plot(X[errorInd, 0], X[errorInd, 1], "r" + c[label])


def plotResultsDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, classifierFunction):
    """Plot training and test prediction for datasets 1, 2, or 3.

    Indicates corect and incorrect label predictions, and plots the
    prediction fields as the background color.
    """

    # Create background meshgrid for plotting label fields
    # Change nx and ny to set the resolution of the fields
    nx = 150
    ny = 150

    xMin = np.min((XTrain[:, 0].min(), XTest[:, 0].min())) - 1
    xMax = np.max((XTrain[:, 0].max(), XTest[:, 0].max())) + 1
    yMin = np.min((XTrain[:, 1].min(), XTest[:, 1].min())) - 1
    yMax = np.max((XTrain[:, 1].max(), XTest[:, 1].max())) + 1

    xi = np.linspace(xMin, xMax, nx)
    yi = np.linspace(yMin, yMax, ny)

    XI, YI = np.meshgrid(xi, yi)

    # Setup data depending on classifier type
    XGrid = np.column_stack((XI.flatten(), YI.flatten()))
    LGrid = classifierFunction(XGrid).reshape((nx, ny))

    # Plot training data
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.imshow(
        LGrid, extent=(xMin, xMax, yMin, yMax), cmap="gray", aspect="auto", origin="lower",
    )
    _plotData(XTrain, LTrain, LPredTrain)
    plt.gca().invert_yaxis()
    plt.title(
        f"Training data results (green ok, red error)" + 
        "\n" + 
        f"Error = {100*np.mean(LTrain!=LPredTrain):.2f}% ({np.sum(LTrain!=LPredTrain)} of {LTrain.shape[0]})"
    )

    # Plot test data
    plt.subplot(1,2,2)
    plt.imshow(
        LGrid, extent=(xMin, xMax, yMin, yMax), cmap="gray", aspect="auto", origin="lower",
    )
    _plotData(XTest, LTest, LPredTest)
    plt.gca().invert_yaxis()
    plt.title(
        f"Test data results (green ok, red error)" + 
        "\n" + 
        f"Error = {100*np.mean(LTest!=LPredTest):.2f}% ({np.sum(LTest!=LPredTest)} of {LTest.shape[0]})"
    )

    # Plot
    plt.show()
    
def plotResultsDotsGradient(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, classifierFunction):
    """Plot training and test prediction for datasets 1, 2, or 3.

    Indicates corect and incorrect label predictions, and plots the
    prediction fields as the background color.
    """

    # Create background meshgrid for plotting label fields
    # Change nx and ny to set the resolution of the fields
    nx = 150
    ny = 150

    xMin = np.min((XTrain[:, 0].min(), XTest[:, 0].min())) - 1
    xMax = np.max((XTrain[:, 0].max(), XTest[:, 0].max())) + 1
    yMin = np.min((XTrain[:, 1].min(), XTest[:, 1].min())) - 1
    yMax = np.max((XTrain[:, 1].max(), XTest[:, 1].max())) + 1

    xi = np.linspace(xMin, xMax, nx)
    yi = np.linspace(yMin, yMax, ny)

    XI, YI = np.meshgrid(xi, yi)

    # Setup data depending on classifier type
    XGrid = np.column_stack((XI.flatten(), YI.flatten()))
    YGrid = classifierFunction(XGrid)
    PGrid = np.exp(YGrid) / np.sum(np.exp(YGrid), axis=1, keepdims=True)
    PGrid = np.clip((PGrid) * 1.6 - 0.5, 0, 1) # For color adjustment
    
    # Plot training data
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    for L in np.unique(LTrain):
        plt.imshow(
            np.ones((nx,ny)), extent=(xMin, xMax, yMin, yMax), cmap=["Reds", "Greens", "Blues"][L], vmin=0, vmax=1,
            aspect="auto", origin="lower", alpha=PGrid[:,L].reshape((nx,ny))
        )
    _plotData(XTrain, LTrain, LPredTrain)
    plt.gca().invert_yaxis()
    plt.title(
        f"Training data results (green ok, red error)" + 
        "\n" + 
        f"Error = {100*np.mean(LTrain!=LPredTrain):.2f}% ({np.sum(LTrain!=LPredTrain)} of {LTrain.shape[0]})"
    )

    # Plot test data
    plt.subplot(1,2,2)
    for L in np.unique(LTest):
        plt.imshow(
            np.ones((nx,ny)), extent=(xMin, xMax, yMin, yMax), cmap=["Reds", "Greens", "Blues"][L], vmin=0, vmax=1,
            aspect="auto", origin="lower", alpha=PGrid[:,L].reshape((nx,ny))
        )
    _plotData(XTest, LTest, LPredTest)
    plt.gca().invert_yaxis()
    plt.title(
        f"Test data results (green ok, red error)" + 
        "\n" + 
        f"Error = {100*np.mean(LTest!=LPredTest):.2f}% ({np.sum(LTest!=LPredTest)} of {LTest.shape[0]})"
    )

    # Plot
    plt.show()


def _plotCase(X, L):
    """Simple plot of data. Can only be used with dataset 1, 2, and 3."""

    plotStrings = ["r.", "g.", "b."]

    for label in range(3):
        ind = (L == label).squeeze()
        plt.plot(X[ind, 0], X[ind, 1], plotStrings[label])
    plt.gca().invert_yaxis()


def plotIsolines(X, L, classifierFunction):
    """Plot isolevels of neural network output for datasets 1-3."""

    cmaps = ["Reds", "Greens", "Blues"]

    # Create background meshgrid for plotting label fields
    # Change nx and ny to set the resolution of the fields
    nx = 150
    ny = 150

    xMin, yMin = X.min(axis=0) - 1
    xMax, yMax = X.max(axis=0) + 1

    xi = np.linspace(xMin, xMax, nx)
    yi = np.linspace(yMin, yMax, ny)

    XI, YI = np.meshgrid(xi, yi)

    # Setup data depending on classifier type
    XGrid = np.column_stack((XI.flatten(), YI.flatten()))
    YGrid = classifierFunction(XGrid)

    # Plot training data
    plt.figure()
    _plotCase(X, L)
    for i in range(YGrid.shape[1]):
        a = YGrid[:, i].reshape((nx, ny))
        plt.contour(XI, YI, a, np.linspace(0, 2, 6), cmap=cmaps[i])
        plt.contour(XI, YI, a, [1], colors="black")
    # plt.gca().invert_yaxis()

    # Plot
    plt.show()


def plotResultsOCR(X, L, LPred):
    """ PLOTRESULTSOCR
    Plots the results using the 4th dataset (OCR). Selects a
    random set of 16 samples each time.
    """

    L = L.astype(int)
    LPred = LPred.astype(int)

    # Create random sort vector
    ord = np.random.permutation(X.shape[0])

    plt.figure(figsize=(6, 6), tight_layout=True)

    # Plot 16 samples
    for n in range(16):
        idx = ord[n]
        plt.subplot(4, 4, n + 1)
        plt.imshow(X[idx].reshape((8, 8)), cmap="gray")
        plt.title("$L_{true}=$" + f"{L[idx]}" + "\n $L_{pred}=$" + f"{LPred[idx]}")
        plt.axis("off")
    plt.suptitle("Random selection of samples")
    plt.show()


def plotConfusionMatrixOCR(X, L, LPred):
    canvas = np.zeros((107, 107))
    for i in range(10):
        for j in range(10):

            I = np.flatnonzero((LPred == i) & (L == j))

            if I.size != 0:
                canvas[i * 11 : i * 11 + 8, j * 11 : j * 11 + 8] = X[np.random.choice(I)].reshape((8, 8))

    plt.figure(figsize=(6,6))
    plt.imshow(canvas, cmap="gray")
    plt.xticks(ticks=np.arange(3, 107, 11), labels=np.arange(10))
    plt.yticks(ticks=np.arange(3, 107, 11), labels=np.arange(10))
    plt.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.xlabel("Actual class")
    plt.ylabel("Predicted class")
    plt.gca().xaxis.set_label_position("top")
    plt.title("Examples cases from the confusion matrix")
    plt.show()


def plotProgress(fig, metrics, n=None):

    numIterations = metrics["lossTrain"].shape[0]
    if n is None:
        n = numIterations

    minErrTest = np.nanmin(metrics["lossTest"][:n])
    minErrTestInd = np.nanargmin(metrics["lossTest"][:n])
    maxAccTest = np.nanmax(metrics["accTest"][:n])
    maxAccTestInd = np.nanargmax(metrics["accTest"][:n])
    
    plt.subplot(2, 1, 1)
    plt.cla()
    plt.semilogy(metrics["lossTrain"][:n], "k", linewidth=1.5, label="Training Loss")
    plt.semilogy(metrics["lossTest"][:n], "r", linewidth=1.5, label="Test Loss")
    plt.semilogy(minErrTestInd, minErrTest, "bo", linewidth=1.5, label="Min Test Loss")

    plt.xlim([0, numIterations])
    plt.grid("on")
    plt.title("Training and Test Losses, Single Layer")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    plt.subplot(2, 1, 2)
    plt.cla()
    plt.plot(metrics["accTrain"][:n], "k", linewidth=1.5, label="Training Accuracy")
    plt.plot(metrics["accTest"][:n], "r", linewidth=1.5, label="Test Accuracy")
    plt.plot(maxAccTestInd, maxAccTest, "bo", linewidth=1.5, label="Max Test Accuracy")

    plt.xlim([0, numIterations])
    plt.grid("on")
    plt.title("Training and Test Accuracies, Single Layer")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    display.display(fig)
    display.clear_output(wait=True)


def plotProgressNetwork(fig, W, B, metrics, cmap="coolwarm", n=None):

    plt.clf()
    plt.subplot(2, 3, (1, 4))
    plt.axis("off")

    # Colormap
    cm = plt.cm.get_cmap(cmap)

    # Normalize and center weights
    W = np.concatenate((W, B), axis=0)
    vmax = np.max([W.max(), -W.min()])
    W2 = W / (2 * vmax) + 0.5

    # y positions of nodes
    nIn, nOut = W.shape
    yIn = np.linspace(-(nIn - 1) / 2, (nIn - 1) / 2, nIn)
    yOut = np.linspace(-(nOut - 1) / 2, (nOut - 1) / 2, nOut)

    # Plot neural network weights
    for i in range(nIn):
        for j in range(nOut):
            plt.plot(
                [0, 1],
                [yIn[i], yOut[j]],
                color=cm(W2[i, j]),
                lw=5,
                marker="o",
                markersize=20,
                markerfacecolor="w",
                markeredgecolor="k",
            )

    # Plot input and output labels
    for i in range(nIn - 1):
        plt.text(-0.1, yIn[i] + 0.03, f"$X_{i}$", fontsize=16)
    plt.text(-0.09, yIn[-1] + 0.03, "1", fontsize=16)

    for j in range(nOut):
        plt.text(1.035, yOut[j] + 0.03, f"$Y_{j}$", fontsize=16)

    plt.title("Network weights")

    # Epoch number
    if n is not None:
        plt.suptitle(f"Epoch {n}")

    # Invert y axis
    ax = plt.gca()
    ax.invert_yaxis()

    # Colorbar
    norm = plt.cm.ScalarMappable(norm=None, cmap=cm)
    norm.set_clim(-vmax, vmax)
    plt.colorbar(norm, ax=ax, location="right")

    ######

    numIterations = metrics["lossTrain"].shape[0]
    if n is None:
        n = numIterations

    minErrTest = np.nanmin(metrics["lossTest"][:n])
    minErrTestInd = np.nanargmin(metrics["lossTest"][:n])
    maxAccTest = np.nanmax(metrics["accTest"][:n])
    maxAccTestInd = np.nanargmax(metrics["accTest"][:n])

    plt.subplot(2, 3, (2, 3))
    # plt.subplot(2,2,2)
    plt.cla()
    plt.semilogy(metrics["lossTrain"][:n], "k", linewidth=1.5, label="Training Loss")
    plt.semilogy(metrics["lossTest"][:n], "r", linewidth=1.5, label="Test Loss")
    plt.semilogy(minErrTestInd, minErrTest, "bo", linewidth=1.5, label="Min Test Loss")

    plt.xlim([0, numIterations])
    plt.grid("on")
    plt.title("Training and Test Losses, Single Layer")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    plt.subplot(2, 3, (5, 6))
    # plt.subplot(2,2,4)
    plt.cla()
    plt.plot(metrics["accTrain"][:n], "k", linewidth=1.5, label="Training Accuracy")
    plt.plot(metrics["accTest"][:n], "r", linewidth=1.5, label="Test Accuracy")
    plt.plot(maxAccTestInd, maxAccTest, "bo", linewidth=1.5, label="Max Test Accuracy")

    plt.xlim([0, numIterations])
    plt.grid("on")
    plt.title("Training and Test Accuracies, Single Layer")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    # display.display(plt.gcf())
    display.display(fig)
    display.clear_output(wait=True)


def plotProgressNetworkMulti(fig, W1, B1, W2, B2, metrics, cmap="coolwarm", n=None):

    plt.clf()
    plt.subplot(2, 3, (1, 4))
    plt.axis("off")

    # Colormap
    cm = plt.cm.get_cmap(cmap)

    # Normalize and center weights
    # W1 = np.concatenate((W1, B1), axis=0)
    # vmax = np.max([W1.max(), -W1.min()])
    # W1 = W1 / (2 * vmax) + 0.5

    # W2 = np.concatenate((W2, B2), axis=0)
    # vmax = np.max([W1.max(), -W2.min()])
    # W2 = W2 / (2 * vmax) + 0.5
    W1 = np.concatenate((W1, B1), axis=0)
    W2 = np.concatenate((W2, B2), axis=0)
    vmax = np.max([W1.max(), W2.max(), -W1.min(), -W2.min()])
    W1 = W1 / (2 * vmax) + 0.5
    W2 = W2 / (2 * vmax) + 0.5

    # y positions of nodes
    nIn1, nOut1 = W1.shape
    yIn1 = np.linspace(-(nIn1 - 1) / 2, (nIn1 - 1) / 2, nIn1)
    yOut1 = np.linspace(-(nOut1 - 1) / 2, (nOut1 - 1) / 2, nOut1)

    nIn2, nOut2 = W2.shape
    yIn2 = np.linspace(-(nIn2 - 1) / 2, (nIn2 - 1) / 2, nIn2)
    yOut2 = np.linspace(-(nOut2 - 1) / 2, (nOut2 - 1) / 2, nOut2)

    # Plot neural network weights
    for i in range(nIn1):
        for j in range(nOut1):
            plt.plot(
                [0, 1],
                [yIn1[i], yOut1[j] - 0.5],
                color=cm(W1[i, j]),
                lw=5,
                marker="o",
                markersize=20,
                markerfacecolor="w",
                markeredgecolor="k",
            )

    for i in range(nIn2):
        for j in range(nOut2):
            plt.plot(
                [1, 2],
                [yIn2[i], yOut2[j]],
                color=cm(W2[i, j]),
                lw=5,
                marker="o",
                markersize=20,
                markerfacecolor="w",
                markeredgecolor="k",
            )

    # # Plot input and output labels
    for i in range(nIn1 - 1):
        plt.text(-0.22, yIn1[i] + 0.03, f"$X_{i}$", fontsize=16)
    plt.text(-0.18, yIn1[-1] + 0.03, "1", fontsize=16)

    for j in range(nIn2 - 1):
        plt.text(0.95, yIn2[j] + 0.52, f"$U_{j}$", fontsize=16)
    plt.text(0.97, yIn2[-1] + 0.52, "1", fontsize=16)

    for k in range(nOut2):
        plt.text(2.1, yOut2[k] + 0.03, f"$Y_{k}$", fontsize=16)

    plt.title("Network weights")

    # Epoch number
    if n is not None:
        plt.suptitle(f"Epoch {n}")

    # Invert y axis
    ax = plt.gca()
    ax.invert_yaxis()

    # Colorbar
    norm = plt.cm.ScalarMappable(norm=None, cmap=cm)
    norm.set_clim(-vmax, vmax)
    plt.colorbar(norm, ax=ax, location="right")

    ######

    numIterations = metrics["lossTrain"].shape[0]
    if n is None:
        n = numIterations

    minErrTest = np.nanmin(metrics["lossTest"][:n])
    minErrTestInd = np.nanargmin(metrics["lossTest"][:n])
    maxAccTest = np.nanmax(metrics["accTest"][:n])
    maxAccTestInd = np.nanargmax(metrics["accTest"][:n])

    plt.subplot(2, 3, (2, 3))
    # plt.subplot(2,2,2)
    plt.cla()
    plt.semilogy(metrics["lossTrain"][:n], "k", linewidth=1.5, label="Training Loss")
    plt.semilogy(metrics["lossTest"][:n], "r", linewidth=1.5, label="Test Loss")
    plt.semilogy(minErrTestInd, minErrTest, "bo", linewidth=1.5, label="Min Test Loss")

    plt.xlim([0, numIterations])
    plt.grid("on")
    plt.title("Training and Test Losses, Single Layer")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    plt.subplot(2, 3, (5, 6))
    # plt.subplot(2,2,4)
    plt.cla()
    plt.plot(metrics["accTrain"][:n], "k", linewidth=1.5, label="Training Accuracy")
    plt.plot(metrics["accTest"][:n], "r", linewidth=1.5, label="Test Accuracy")
    plt.plot(maxAccTestInd, maxAccTest, "bo", linewidth=1.5, label="Max Test Accuracy")

    plt.xlim([0, numIterations])
    plt.grid("on")
    plt.title("Training and Test Accuracies, Single Layer")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    # display.display(plt.gcf())
    display.display(fig)
    display.clear_output(wait=True)


def plotProgressOCR(fig, W, metrics, cmap="coolwarm", n=None):

    # w = W[:-1,i].reshape(8,8)
    vmax = np.max([W.max(), -W.min()])
    vmin = -vmax

    for i in range(10):
        w = W[:, i].reshape(8, 8)
        # vmax = np.max([w.max(), -w.min()])
        # vmin = -vmax

        plt.subplot(2, 10, i + 1 + 5 * (i // 5))
        plt.cla()
        plt.axis("off")
        plt.imshow(w, vmin=vmin * 0.8, vmax=vmax * 0.8, cmap=cmap)
        plt.title(i)

    plt.subplot(2, 10, 3)
    plt.title("Network weights for each digit (blue: positive, red: negative) \n\n 2")

    # plt.subplot(2,10,1)
    # plt.colorbar(location="left")

    if n is not None:
        plt.suptitle(f"Epoch {n}")

    ######

    numIterations = metrics["lossTrain"].shape[0]
    if n is None:
        n = numIterations

    minErrTest = np.nanmin(metrics["lossTest"][:n])
    minErrTestInd = np.nanargmin(metrics["lossTest"][:n])
    maxAccTest = np.nanmax(metrics["accTest"][:n])
    maxAccTestInd = np.nanargmax(metrics["accTest"][:n])

    plt.subplot(2, 10, (6, 10))
    plt.cla()
    plt.semilogy(metrics["lossTrain"][:n], "k", linewidth=1.5, label="Training Loss")
    plt.semilogy(metrics["lossTest"][:n], "r", linewidth=1.5, label="Test Loss")
    plt.semilogy(minErrTestInd, minErrTest, "bo", linewidth=1.5, label="Min Test Loss")

    plt.xlim([0, numIterations])
    plt.grid("on")
    plt.title("Training and Test Losses, Single Layer")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    plt.subplot(2, 10, (16, 20))
    plt.cla()
    plt.plot(metrics["accTrain"][:n], "k", linewidth=1.5, label="Training Accuracy")
    plt.plot(metrics["accTest"][:n], "r", linewidth=1.5, label="Test Accuracy")
    plt.plot(maxAccTestInd, maxAccTest, "bo", linewidth=1.5, label="Max Test Accuracy")

    plt.xlim([0, numIterations])
    plt.grid("on")
    plt.title("Training and Test Accuracies, Single Layer")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    #####

    display.display(fig)
    display.clear_output(wait=True)