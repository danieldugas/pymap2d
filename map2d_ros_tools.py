import rospy
import tf
from threading import Lock
from CMap2D import CMap2D

def default_refmap_update_callback(self_):
    pass

class ReferenceMapAndLocalizationManager(object):
    """ If a reference map is provided and a tf exists,
    keeps track of the tf for given frame in the reference map frame """
    def __init__(self, map_folder, map_filename, reference_map_frame, frame_to_track, refmap_update_callback=default_refmap_update_callback):
        self.tf_frame_in_refmap = None
        self.map_ = None
        self.lock = Lock() # for avoiding race conditions
        self.refmap_is_dynamic = False
        # update callback
        self.refmap_update_callback = refmap_update_callback
        # get frame info
        self.kRefMapFrame = reference_map_frame
        self.kFrame = frame_to_track
        # Load map
        if map_folder == "rostopic":
            self.refmap_is_dynamic = True
            rospy.logwarn("Getting reference map from topic '{}'".format(map_filename))
            from nav_msgs.msg import OccupancyGrid
            rospy.Subscriber(map_filename, OccupancyGrid, self.map_callback, queue_size=1)
        else:
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
            self.refmap_update_callback(self)
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

    def map_callback(self, msg):
        with self.lock:
            self.map_ = CMap2D()
            self.map_.from_msg(msg)
            self.refmap_update_callback(self)

