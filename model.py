import logging
import numpy
import theano
logger = logging.getLogger(__name__)

class Model(object):
    def __init__(self):
        self.floatX = theano.config.floatX
        # Parameters of the model
        self.params = []
    
    def save(self, filename):
        """
        Save the model to file `filename`
        """
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(filename, **vals)

    def load(self, filename):
        """
        Load the model.
        """
        vals = numpy.load(filename)
        for p in self.params:
            if p.name in vals:
                logger.debug('Loading {} of {}'.format(p.name, p.get_value(borrow=True).shape))
                if p.get_value().shape != vals[p.name].shape:
                    # Trick to initialize conditioned RNN from unconditioned RNN. 
                    # We just need to fix the softmax matrices, by broadcasting them along their first dimension.
                    if ('softmax' in p.name) and (p.get_value().shape[0] == 6) and (len(vals[p.name].shape) + 1 == len(p.get_value().shape)):
                        print 'Adjusted softmax for parameter: ', str(p.name)
                        broadcasted_parameter = numpy.zeros(p.get_value().shape, dtype=self.floatX)
                        for i in range(6):
                            if len(p.get_value().shape) == 2:
                                broadcasted_parameter[i, :] = numpy.squeeze(vals[p.name])
                            elif len(p.get_value().shape) == 3:
                                broadcasted_parameter[i, :, :] = vals[p.name]
                            else:
                                print 'Error!'
                        broadcasted_parameter = broadcasted_parameter.astype(self.floatX)
                        print 'Broadcasted_parameter', broadcasted_parameter.shape, broadcasted_parameter
                        p.set_value(broadcasted_parameter)
                    else:
                        raise Exception('Shape mismatch: {} != {} for {}'.format(p.get_value().shape, vals[p.name].shape, p.name))
                else:
                    p.set_value(vals[p.name])
            else:
                logger.error('No parameter {} given: default initialization used'.format(p.name))
                unknown = set(vals.keys()) - {p.name for p in self.params}
                if len(unknown):
                    logger.error('Unknown parameters {} given'.format(unknown))
