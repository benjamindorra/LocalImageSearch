#!/usr/local/bin/python3

import PyQt6.QtCore as qtc
from PyQt6.QtGui import QPalette, QPixmap, QDrag
import PyQt6.QtWidgets as qtw
from dbSearch import get_df, get_imgs, searchText, searchImg, getFromTo, getNumChunks
import dbIndex
import traceback
import sys
import os
import pandas as pd
from config import ENCODED_IMAGES_PATH, MODEL_PATH, PRETRAINED_ENCODED_IMAGES_PATH, PRETRAINED_MODEL_PATH
from config import INDEXING_DIR, MATCHING_FILE
from config import SETTINGS_PATH
import json
from PIL import Image

def get_df_embpath(model_path, imgs_dir_path):
  matching_file = os.path.join(os.path.dirname(model_path), MATCHING_FILE)
  matching_df = pd.read_csv(matching_file)
  imgs_dir_row = matching_df[matching_df["images"]==imgs_dir_path]
  infos_path = imgs_dir_row["infos"].values[0]
  embeddings_path = imgs_dir_row["embeddings"].values[0]
  df = pd.read_csv(infos_path)
  return df, embeddings_path

def get_model_path():
  if modelChoice.currentText() == 'Pretrained':
    model_path = PRETRAINED_MODEL_PATH
  else:
    model_path = MODEL_PATH
  return model_path


class Search:
  """Main class for interacting with the database"""
  def __init__(self, pageSize):
    self.pageSize = pageSize
    self.values = None

  def imageSearch(self, result):
    """Search by image and store result index"""
    self.imgDir = dirChoice.pathBar.text()
    self.values = result

  def textSearch(self, request):
    """Search by text and store result index"""
    self.imgDir = dirChoice.pathBar.text()
    self.imgDir,self.imgNames,self.imgPaths = get_imgs(self.imgDir)
    category = self.catChoice.currentText()
    category = None if category=='All' else category
    self.values = searchText(request,self.df,self.imgDir,category)

  def getPage(self, page):
    """Get a page to display from the db, starting from 1"""
    start = (page - 1) * self.pageSize.value()
    end = start + self.pageSize.value()
    return getFromTo(self.values, self.imgDir, start, end)

  def getNumPages(self):
    return getNumChunks(self.values, self.pageSize.value())


#https://stackoverflow.com/questions/50232639/drag-and-drop-qlabels-with-pyqt5
#https://stackoverflow.com/questions/64252654/pyqt5-drag-and-drop-into-system-file-explorer-with-delayed-encoding
class DragLabel(qtw.QLabel):
    def __init__(self, text, parent):
        super(DragLabel, self).__init__(parent)
        self.url = [qtc.QUrl.fromLocalFile(text)]

    def mousePressEvent(self, event):
        #super().mousePressEvent(event)
        self.dragStartPos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() != qtc.Qt.MouseButton.LeftButton:
            return
        if (event.pos() - self.dragStartPos).manhattanLength()\
        < qtw.QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mimedata = qtc.QMimeData()
        mimedata.setUrls(self.url)

        drag.setMimeData(mimedata)
        drag.setPixmap(self.grab().scaledToHeight(100))
        drag.exec(qtc.Qt.DropAction.CopyAction)

class NavBar(qtw.QWidget):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.parent = parent
    # Navigation
    self.navLayout = qtw.QHBoxLayout()
    self.navPrevButton = qtw.QPushButton("Prev")
    self.navPrevButton.clicked.connect(parent.prevPage)
    self.navPrevButton.setFixedWidth(self.navPrevButton.minimumSizeHint().width())
    self.navLayout.addWidget(self.navPrevButton)
    self.navSelectPage = qtw.QLineEdit("")
    self.navSelectPage.returnPressed.connect(self.selectPage)
    self.navLayout.addWidget(self.navSelectPage)
    self.navLabel = qtw.QLabel("/"+"")
    self.navLayout.addWidget(self.navLabel)
    self.navNextButton = qtw.QPushButton("Next")
    self.navNextButton.clicked.connect(parent.nextPage)
    self.navNextButton.setFixedWidth(self.navNextButton.minimumSizeHint().width())
    self.navLayout.addWidget(self.navNextButton)
    self.navLayout.setAlignment(qtc.Qt.AlignmentFlag.AlignHCenter)

  def selectPage(self):
    pageRequest = self.navSelectPage.text()
    self.parent.selectPage(pageRequest)


class customSubWindow(qtw.QWidget):
  """Subwindow to show the results of a search"""
  def __init__(self,caller,wid,searchModule,parent=None):
    super(customSubWindow,self).__init__(parent)
    # Parameters
    self.caller = caller
    self.wid = wid
    self.searchModule = searchModule
    # Window size
    # Main widgets in scroll area
    self.scrollLayout = qtw.QVBoxLayout()
    self.scrollLayout.setSizeConstraint(self.scrollLayout.SizeConstraint.SetMinAndMaxSize)
    self.displayArea = qtw.QWidget(self)
    self.displayArea.setLayout(self.scrollLayout)
    # Scrolling
    self.scrollArea = qtw.QScrollArea(self)
    self.scrollArea.setWidget(self.displayArea)
    self.scrollArea.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    self.subLayout = qtw.QVBoxLayout()
    self.subLayout.addWidget(self.scrollArea)
    self.setLayout(self.subLayout)
    # Objects on the page
    self.lines = []
    self.infos = []
    self.images = []
    self.titles = []
    # Navigation bars
    self.topNavBar = NavBar(self)
    self.botNavBar = NavBar(self)
    self.scrollLayout.addLayout(self.topNavBar.navLayout)

  def start(self):
    self.finalPage = self.searchModule.getNumPages()
    self.topNavBar.navSelectPage.setFixedWidth(self.topNavBar.navSelectPage.minimumSizeHint().width())
    self.botNavBar.navSelectPage.setFixedWidth(self.botNavBar.navSelectPage.minimumSizeHint().width())
    if self.finalPage>0:
      firstPage = self.searchModule.getPage(1)
    else:
      # No results
      firstPage = self.searchModule.values
    self.currentPage = 1
    self.updateNavBars()
    self.topNavBar.navLabel.setFixedWidth(self.topNavBar.navLabel.minimumSizeHint().width())
    self.botNavBar.navLabel.setFixedWidth(self.botNavBar.navLabel.minimumSizeHint().width())
    self.displayResults(firstPage)
    self.show()

  def cleanPage(self):
    for i,line in enumerate(self.lines):
      line.removeWidget(self.images[i])
      line.removeWidget(self.infos[i])
      self.images[i].deleteLater()
      self.infos[i].deleteLater()
      self.images[i] = None
      self.infos[i] = None
      self.scrollLayout.removeItem(line)
      line.deleteLater()
      line = None
    for title in self.titles:
      self.scrollLayout.removeWidget(title)
      title.deleteLater()
      title = None
    self.lines = []
    self.infos = []
    self.images = []
    self.titles = []
    self.scrollLayout.removeItem(self.botNavBar.navLayout)


  def closeEvent(self, event):
    '''Redefined to close all children widgets and save settings'''
    #Close widgets
    #https://stackoverflow.com/questions/5899826/pyqt-how-to-remove-a-widget
    self.cleanPage()
    self.displayArea.deleteLater()
    self.scrollLayout.deleteLater()
    self.scrollArea.deleteLater()
    self.subLayout.deleteLater()
    self.deleteLater()
    self.caller.delSubWindow(self.wid)
    #Save settings
    settings["search_dir"] = dirChoice.pathBar.text()
    settings["model"] = modelChoice.currentText()
    settings["page_size"] = pageSize.value() 
    with open(SETTINGS_PATH, "w") as f:
      json.dump(settings, f, indent=4)
    #Close window
    event.accept()

  def displayResults(self,results):    
    for category in results:
      #self.scrollLayout.addWidget(self.titles[-1])
      for entry in category[1]:
        self.lines.append(qtw.QVBoxLayout())
        self.scrollLayout.addLayout(self.lines[-1])
        self.infos.append(qtw.QLabel(entry[0],self))
        self.infos[-1].setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.images.append(DragLabel(entry[1],self))
        width = 512
        self.images[-1].setPixmap(QPixmap(entry[1]).scaledToWidth(width))
        self.lines[-1].addWidget(self.infos[-1])
        self.lines[-1].addWidget(self.images[-1])
    self.scrollLayout.addLayout(self.botNavBar.navLayout)

  def updateNavBars(self):
    """Update displayed page number"""
    self.topNavBar.navLabel.setText("/"+str(self.finalPage))
    self.botNavBar.navLabel.setText("/"+str(self.finalPage))
    self.topNavBar.navSelectPage.setText(str(self.currentPage))
    self.botNavBar.navSelectPage.setText(str(self.currentPage))

  def prevPage(self):
    if self.currentPage > 1:
      self.currentPage -= 1
      self.updateNavBars()
      self.showPage()

  def nextPage(self):
    if self.currentPage < self.finalPage:
      self.currentPage += 1
      self.updateNavBars()
      self.showPage()

  def selectPage(self, pageRequest):
    try:
      page = int(pageRequest)
    except ValueError:
      return
    if page>=1 and page != self.currentPage and page <= self.finalPage:
      self.currentPage = page
      self.updateNavBars()
      self.showPage()

  def showPage(self):
    self.cleanPage()
    self.scrollArea.verticalScrollBar().setValue(0)
    content = self.searchModule.getPage(self.currentPage)
    self.displayResults(content)

#Signals to manage multiprocess
#https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/
class ImSearchWorkerSignals(qtc.QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    '''
    finished = qtc.pyqtSignal(int)
    error = qtc.pyqtSignal(tuple)
    result = qtc.pyqtSignal(object)
    updateBar = qtc.pyqtSignal(int)


#Image search worker
class ImSearchWorker(qtc.QRunnable):
  '''
  Worker thread
  '''
  def __init__(self, *args, **kwargs):
    super(ImSearchWorker, self).__init__()
    self.args = args
    self.kwargs = kwargs
    self.signals = ImSearchWorkerSignals()

  @qtc.pyqtSlot()
  def run(self):
    '''
    Your code goes in this function
    '''
    try:
      returnValue = searchImg(*self.args,
                              signal=self.signals.updateBar,tid=self.kwargs['tid'])
    except:
      traceback.print_exc()
      exctype, value = sys.exc_info()[:2]
      self.signals.error.emit((exctype, value, traceback.format_exc()))
    else:
      self.signals.result.emit(returnValue)  # Return the result of the processing
    finally:
      self.signals.finished.emit(self.kwargs['tid'])  # Done

#Zone for drag and drop
class DragDrop(qtw.QFrame):
  def __init__(self, modelChoice, pageSize, parent=None):
    super(DragDrop,self).__init__(parent)

    #Text zone setup
    self.setAcceptDrops(True)
    self.setStyleSheet("background-color: grey;")
    self.setToolTip(
      'Drag and drop an image to search for it (may take a few minutes)'
    )

    #Get parameters
    self.modelChoice = modelChoice
    self.pageSize = pageSize
    
    #Thread setup
    self.threadPool = qtc.QThreadPool()
    self.progressBars = []

    #Keep subwindows in scope
    self.subWindows = []


  def dragEnterEvent(self,event):
    if event.mimeData().hasUrls():
      self.setStyleSheet("background-color: lightgrey;")
      event.accept()
    else:
      event.ignore()

  def dragLeaveEvent(self,event):
    self.setStyleSheet("background-color: grey;")
    event.accept()

  def showOutput(self,result):
    searchModule = Search(self.pageSize)
    searchModule.imageSearch(result)
    self.subWindows.append(customSubWindow(self,len(self.subWindows),searchModule))
    self.subWindows[-1].start()

  def delSubWindow(self,wid):
    self.subWindows[wid].deleteLater()
    self.subWindows[wid]=None

  def threadComplete(self,tid):
    #Remove progress bar for completed process
    #https://stackoverflow.com/questions/5899826/pyqt-how-to-remove-a-widget
    self.parent().layout().removeWidget(self.progressBars[tid])
    self.progressBars[tid].deleteLater()
    self.progressBars[tid] = None

  def updateBar(self,tid):
    self.progressBars[tid].setValue(self.progressBars[tid].value()+1)

  def dropEvent(self,event):
    self.setStyleSheet("background-color: grey;")
    imgDir = dirChoice.pathBar.text()
    _,self.imgNames,self.imgPaths = get_imgs(imgDir)
    self.numImages = len(self.imgNames)-1
    if self.modelChoice.currentText() == 'Pretrained':
        model_path = PRETRAINED_MODEL_PATH
    else:
        model_path = MODEL_PATH

    files = [u.toLocalFile() for u in event.mimeData().urls()]

    matching_file = os.path.join(os.path.dirname(model_path), MATCHING_FILE)
    matching_df = pd.read_csv(matching_file)

    if not matching_df["images"].str.fullmatch(dirChoice.pathBar.text()).any():
      reply = qtw.QMessageBox.question(self, 'No index', 'No index for this directory, do you want to index it (this may take a while) ?',
      qtw.QMessageBox.StandardButton.Yes | qtw.QMessageBox.StandardButton.No, qtw.QMessageBox.StandardButton.No)
      if reply == qtw.QMessageBox.StandardButton.No:
        return
      if reply == qtw.QMessageBox.StandardButton.Yes:
        self.indexer = Index(images_dir=dirChoice.pathBar.text(), encoder_path=model_path, parent=window)
        self.indexer()
    else:
        self.img_search(model_path, event, files)


  def img_search(self, model_path, event, files):
    imgs_dir_path = dirChoice.pathBar.text()
    self.df, embeddings_path = get_df_embpath(get_model_path(), imgs_dir_path)

    for f in files:
      #Progress bar
      self.progressBars.append(qtw.QProgressBar(self.parent()))
      self.progressBars[-1].setMinimum(0)
      self.progressBars[-1].setMaximum(self.numImages)
      self.progressBars[-1].setToolTip('Progress bar for image search')
      self.parent().layout().addWidget(self.progressBars[-1])

      #start search
      tid = len(self.progressBars)-1
      worker = ImSearchWorker(f,self.df,model_path,embeddings_path,tid=tid)
      #Account for signals
      worker.signals.finished.connect(self.threadComplete)
      worker.signals.result.connect(self.showOutput)
      worker.signals.updateBar.connect(self.updateBar)
      #Start process
      self.threadPool.start(worker)
    event.accept()

#Search bar
class searchBar(qtw.QLineEdit):
  def __init__(self,languageChoice,catChoice,pageSize,parent=None,placeholderText=None):
    super(searchBar,self).__init__(parent)
    self.setPlaceholderText(placeholderText)
    self.setToolTip("term to search (no regex)")
    self.returnPressed.connect(self.startSearch)
    self.catChoice = catChoice
    self.pageSize = pageSize
    imgDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    self.imgDir,_,_ = get_imgs(imgDir)
    self.subWindows = []

  def startSearch(self):
    searchModule = Search(languageChoice, catChoice, pageSize)
    searchModule.textSearch(self.text())
    self.subWindows.append(customSubWindow(self,len(self.subWindows),searchModule))
    self.subWindows[-1].start()

  def delSubWindow(self,wid):
    self.subWindows[wid].deleteLater()
    self.subWindows[wid]=None

#Signals to manage multiprocess
#https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/
class IndexingWorkerSignals(qtc.QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    updateBar
        No data


    '''
    finished = qtc.pyqtSignal()
    error = qtc.pyqtSignal(tuple)
    updateBar = qtc.pyqtSignal()

#Indexing worker
class IndexingWorker(qtc.QRunnable):
  '''
  Worker thread
  '''
  def __init__(self, *args, **kwargs):
    super(IndexingWorker, self).__init__()
    self.args = args
    self.kwargs = kwargs
    self.signals = IndexingWorkerSignals()

  @qtc.pyqtSlot()
  def run(self):
    '''
    Your code goes in this function
    '''
    try:
      dbIndex.indexDir(
              *self.args,
              signal=self.signals.updateBar,
              )
    except:
      traceback.print_exc()
      exctype, value = sys.exc_info()[:2]
      self.signals.error.emit((exctype, value, traceback.format_exc()))
    finally:
      self.signals.finished.emit()  # Done


class Index():
  def __init__(self, images_dir, encoder_path, parent=None):
    #Parameters
    self.parent = parent
    self.images_dir = images_dir
    self.encoder_path = encoder_path
    #Threads
    self.threadPool = qtc.QThreadPool()
    #Progress bar
    self.num_images = len(os.listdir(self.images_dir))
    self.progressBar = qtw.QProgressBar(self.parent)
    self.progressBar.setMinimum(0)
    self.progressBar.setMaximum(self.num_images)
    self.progressBar.setToolTip('Progress bar for directory indexing')
    self.parent.layout().addWidget(self.progressBar)
    
  def __call__(self):
    #start indexing
    worker = IndexingWorker(self.images_dir,self.encoder_path)
    #Account for signals
    worker.signals.finished.connect(self.threadComplete)
    #worker.signals.result.connect(self.showOutput)
    worker.signals.updateBar.connect(self.updateBar)
    #Start process
    self.threadPool.start(worker)

  def threadComplete(self):
    #Remove progress bar for completed process
    #https://stackoverflow.com/questions/5899826/pyqt-how-to-remove-a-widget
    self.parent.layout().removeWidget(self.progressBar)
    self.progressBar.deleteLater()
    self.progressBar = None

  def updateBar(self):
    self.progressBar.setValue(self.progressBar.value()+1)



#Search directory selection
class DirChoice(qtw.QFileDialog):
  def __init__(self, parent=None):
    super().__init__()
    self.pathBar = qtw.QLineEdit(parent)
    self.pathBar.setToolTip("Search directory")
    self.pathBar.setText(settings["search_dir"])
    self.dialogButton = qtw.QPushButton(text="Search in:",parent=window)
    self.dialogButton.clicked.connect(self.getExistingDirectory)
    self.reindexButton = qtw.QPushButton(text="Reindex",parent=window)
    self.reindexButton.clicked.connect(self.reindex)

  
  def getExistingDirectory(self):
    path = super().getExistingDirectory()
    self.pathBar.setText(path)
    #Save settings
    settings["search_dir"] = self.pathBar.text()
    settings["model"] = modelChoice.currentText()
    settings["page_size"] = pageSize.value() 
    with open(SETTINGS_PATH, "w") as f:
      json.dump(settings, f, indent=4)

    return self.pathBar.text()

  def reindex(self):
    if modelChoice.currentText() == 'Pretrained':
        model_path = PRETRAINED_MODEL_PATH
    else:
        model_path = MODEL_PATH
    matching_file = os.path.join(os.path.dirname(model_path), MATCHING_FILE)
    matching_df = pd.read_csv(matching_file)
    execute_reindex = True
    if matching_df["images"].str.fullmatch(self.pathBar.text()).any():
      reply = qtw.QMessageBox.question(self, 'Reindex', 'An index is already defined for this directory, are you sure you want to reindex (this may take a while) ?',
      qtw.QMessageBox.StandardButton.Yes | qtw.QMessageBox.StandardButton.No, qtw.QMessageBox.StandardButton.No)
      if reply == qtw.QMessageBox.StandardButton.No:
        execute_reindex = False
    if execute_reindex:
      drop_index = matching_df.loc[matching_df["images"]==self.pathBar.text()].index
      self.indexer = Index(images_dir=dirChoice.pathBar.text(), encoder_path=model_path, parent=window)
      self.indexer()

#Load settings
with open(SETTINGS_PATH, "r") as f:
  settings = json.loads(f.read())

#Create main window
app = qtw.QApplication([])
app.setStyle('macos')
window = qtw.QWidget()
layout = qtw.QVBoxLayout()
layoutLine0 = qtw.QHBoxLayout()
layoutLine1 = qtw.QHBoxLayout()
layoutLine2 = qtw.QHBoxLayout()
layoutLine3 = qtw.QHBoxLayout()
window.setLayout(layout)
window.setWindowTitle("Search in database")
layout.addLayout(layoutLine0)
layout.addLayout(layoutLine1)
layout.addLayout(layoutLine2)
layout.addLayout(layoutLine3)

#Setup widgets
dirChoice = DirChoice(window)
catOptions = ['All','Index', 'Registration numbers of object', 'Identifier', 'Author', 'Material',
       'Technique', 'Dimensions', 'Acquisition method', 'Item name',
       'Date of origin', 'Place of origin', 'Date of birth']
modelChoice = qtw.QComboBox(window)
modelChoice.addItems(['Pretrained', 'Finetuned'])
modelChoice.setToolTip('Model for reverse image search')
modelChoice.setCurrentIndex(modelChoice.findText(settings["model"]))
pageSize = qtw.QSpinBox(window)
pageSize.setValue(settings["page_size"])
pageSize.setToolTip("Number of images on each page")
dragDrop = DragDrop(modelChoice, pageSize,window)

#Place widgets
window.resize(450,500)
layoutLine0.addWidget(dirChoice.dialogButton)
layoutLine0.addWidget(dirChoice.pathBar)
layoutLine0.addWidget(dirChoice.reindexButton)
layoutLine1.addWidget(modelChoice)
layoutLine1.addWidget(pageSize)
layoutLine3.addWidget(dragDrop)
window.show()
app.exec()
