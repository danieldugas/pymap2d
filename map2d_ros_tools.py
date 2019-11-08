import rospy
import tf
from CMap2D import CMap2D

class ReferenceMapAndLocalizationManager(object):
    """ If a reference map is provided and a tf exists,
    keeps track of the tf for given frame in the reference map frame """
    def __init__(self, map_folder, map_filename, reference_map_frame, frame_to_track):
        self.tf_frame_in_refmap = None
        self.map_ = None
        # loads map based on ros params
        folder = map_folder
        filename =  map_filename
        try:
            self.map_ = CMap2D(folder, filename)
        except IOError as e:
            rospy.logwarn(rospy.get_namespace())
            rospy.logwarn("Failed to load reference map. Make sure {}.yaml and {}.pgm"
                   " are in the {} folder.".format(filename, filename, folder))
            rospy.logwarn("Disabling. No global localization or reference map will be available.")
            return
        # get frame info
        self.kRefMapFrame = reference_map_frame
        self.kFrame = frame_to_track
        # launch callbacks
        self.tf_listener = tf.TransformListener()
        rospy.Timer(rospy.Duration(0.01), self.tf_callback)
        self.has_last_failed = False

    def tf_callback(self, event=None):
        try:
             self.tf_frame_in_refmap = self.tf_listener.lookupTransform(self.kRefMapFrame, self.kFrame, rospy.Time(0))
             if self.has_last_failed:
                 rospy.logwarn("refmap tf found.")
                 self.has_last_failed = False
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(10., e)
            self.has_last_failed = True
            return
