from sklearn.feature_extraction.text import TfidfVectorizer # document vectorization and TF-IDF
from sklearn.datasets import fetch_20newsgroups # test dataset
from sklearn.decomposition import TruncatedSVD # for features reduction
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import pandas as pd





class HandlerCreator:
    """
    Factory method for DataHandler creation

    Use Handler.create_data_handler(test=True) to get a test DataHandler instance
    """
    @staticmethod
    def create_data_handler(test=False):
        if test:
            return DataHandlerTest()
        return DataHandler()





class IDataHandler:
    """
    Interface for data handling
    """
    def __init__(self):
        pass


    def get_docterm_matrix(self):
        raise NotImplementedError





class DataHandlerTest(IDataHandler):
    """
    Concrete DataHandler class, for test purposes and development only

    Public methods:

        - get_docterm_matrix : get the (document, term) matrix, reduced by SVD method. This matrix is a dense numpy.ndarray
    """
    def __init__(self):
        pass


    def _import_data(self):
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
        ]
        self.dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)


    def _vectorize(self):
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=30000, min_df=2, stop_words='english', use_idf=True)
        return vectorizer.fit_transform(self.dataset.data)


    def _reduce_features(self, docterm_mat):
        svd = TruncatedSVD(n_components=200)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        return lsa.fit_transform(docterm_mat)


    def get_docterm_matrix(self):
        """
        Return the reduced document-term matrix. Each row of the matrix represents a document vector
        """
        self._import_data()
        docterm_mat = self._vectorize()
        return self._reduce_features(docterm_mat)





class DataHandler(IDataHandler):
    """
    Concrete DataHandler class, currently in development
    """
    def __init__(self):
        self.dataset = None
        self.docterm_mat = None


    def _vectorize(self):
        vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, use_idf=True)
        return vectorizer.fit_transform(self.dataset['Text'])
    

    def load_dataset(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)

    
    def set_dataset(self, dataset):
        self.dataset = pd.DataFrame(dataset)


    def get_dataset(self):
        return self.dataset


    def get_docterm_matrix(self):
        if self.docterm_mat == None:
            self.docterm_mat = self._vectorize()
        return self.docterm_mat


    def set_docterm_mat(self, docterm_mat):
        self.docterm_mat = docterm_mat



# data_handler = HandlerCreator.create_data_handler(test=True)
# print(data_handler.get_docterm_matrix())


