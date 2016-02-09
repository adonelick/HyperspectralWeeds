from spectronon.workbench.plugin import CubePlugin
from resonon.utils.spec import SpecFloat
from resonon.constants import INTERFACES

class MyPlugin(CubePlugin):
    """
    An example plugin
    """
    label = "My Plugin"
    userLevel = 1

    def setup(self):
         self.my_float = SpecFloat(label='My Float', minval=0, maxval=100,
                                   stepsize=0.1, defaultValue=10.4)
         self.my_float.units = 'my units'
         self.my_float.help = 'my mouse hover help'

         # INTERFACES.SPIN is also valid.  This line is not needed if you
         # just want default behaviour.
         self.my_float.interfaceType = INTERFACES.SLIDER

    def action(self):
        bands = self.datacube.getBandCount()
        samples = self.datacube.getSampleCount()
        lines = self.datacube.getLineCount()

        self.wb.postMessage("Bands: " + str(bands))
        self.wb.postMessage("Samples: " + str(samples))
        self.wb.postMessage("Lines: " + str(lines))