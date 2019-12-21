from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout,QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

from libs.utils import addActions, newAction
from functools import partial
import os

import numpy as np
import pandas as pd

class TableWindow(QMainWindow):
    def __init__(self, parent=None, data=None, save_dir='', col_labels=None, row_labels=None, title=''):
        super(TableWindow, self).__init__(parent)
        self.data = np.array(data)
        self.col_labels = col_labels
        self.row_labels = row_labels
        self.save_dir, self.filename = os.path.split(save_dir)
        self.filename = self.filename.replace(".mjpeg", ".txt")[:-4]
        self.title = title
        self.left = 0
        self.top = 0
        self.width = 300
        self.height = 200
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.createTable()
        self.setCentralWidget(self.tableWidget)
        export = self.menuBar().addMenu("Export")
        action = partial(newAction, self)
        clipboard = action(('Copy'), self.to_clipboard, 'Ctrl+C', 'copy', 'Copy to Clipboard')
        excel = action(('Excel (xls)'), self.to_excel, '', 'xls', 'Export to excel', enabled=self.data.shape[1]<256)
        csv = action(('CSV (csv)'), self.to_csv, '', 'csv', 'Export to csv')
        addActions(export, [clipboard, excel, csv])
        

    def createTable(self):
       # Create table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(self.data.shape[0])
        self.tableWidget.setColumnCount(self.data.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(self.col_labels)
        self.tableWidget.setVerticalHeaderLabels(self.row_labels)
        for i, row in enumerate(self.data):
            for j, val in enumerate(row):
                self.tableWidget.setItem(i,j, QTableWidgetItem(str(val)))
        self.tableWidget.move(0,0)
    
    def to_excel(self):
        path = self.saveFileDialog('.xls')
        df = pd.DataFrame(self.data, index=self.row_labels, columns=self.col_labels)
        df.to_excel(path)
    
    def to_csv(self):
        path = self.saveFileDialog('.csv')
        df = pd.DataFrame(self.data, index=self.row_labels, columns=self.col_labels)
        df.to_csv(path)

    def to_clipboard(self):
        df = pd.DataFrame(self.data, index=self.row_labels, columns=self.col_labels)
        df.to_clipboard()

    def saveFileDialog(self, ext=".txt"):
        caption = 'Choose File'
        filters = 'File (*%s)' % ext
        openDialogPath = self.save_dir
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(ext[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.selectFile(self.filename+'_'+self.title)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            return dlg.selectedFiles()[0]
        return ''