import weakref
import numpy as np

from collections.abc import Callable

import carb
import omni

from ..device_base import Device


class BiSe3Keyboard(Device):
    """A keyboard controller for sending SE(3) commands as delta poses for dual-arm lerobot.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Left Arm Joint 1 (shoulder_pan) Q                 U
        Left Arm Joint 2 (shoulder_lift) W                 I
        Left Arm Joint 3 (elbow_flex)   E                 O
        Left Arm Joint 4 (wrist_flex)   A                 J
        Left Arm Joint 5 (wrist_roll)   S                 K
        Left Arm Joint 6 (gripper)      D                 L
        Right Arm Joint 1 (shoulder_pan) T                 Y
        Right Arm Joint 2 (shoulder_lift) R                 F
        Right Arm Joint 3 (elbow_flex)   G                 H
        Right Arm Joint 4 (wrist_flex)   Z                 X
        Right Arm Joint 5 (wrist_roll)   C                 V
        Right Arm Joint 6 (gripper)      B                 N
        ============================== ================= =================

    Recording Controls:
        ============================== =================
        Description                    Key
        ============================== =================
        Start Recording                SPACE
        End Recording (Task Failed)    P
        End Recording (Task Success)   M
        ============================== =================

    """

    def __init__(self, env, sensitivity: float = 0.05):
        super().__init__(env)
        """Initialize the keyboard layer for dual-arm control.
        """
        # store inputs
        self.sensitivity = sensitivity

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()

        # command buffers for both arms
        self._delta_pos_left = np.zeros(6)
        self._delta_pos_right = np.zeros(6)

        # some flags and callbacks
        self.started = False
        self._reset_state = 0
        self._additional_callbacks = {}

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of dual-arm keyboard."""
        msg = "Dual-Arm Keyboard Controller for SE(3).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tLeft Arm Controls:\n"
        msg += "\t  Joint 1 (shoulder_pan):  Q/U\n"
        msg += "\t  Joint 2 (shoulder_lift): W/I\n"
        msg += "\t  Joint 3 (elbow_flex):    E/O\n"
        msg += "\t  Joint 4 (wrist_flex):    A/J\n"
        msg += "\t  Joint 5 (wrist_roll):    S/K\n"
        msg += "\t  Joint 6 (gripper):       D/L\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight Arm Controls:\n"
        msg += "\t  Joint 1 (shoulder_pan):  T/Y\n"
        msg += "\t  Joint 2 (shoulder_lift): R/F\n"
        msg += "\t  Joint 3 (elbow_flex):    G/H\n"
        msg += "\t  Joint 4 (wrist_flex):    Z/X\n"
        msg += "\t  Joint 5 (wrist_roll):    C/V\n"
        msg += "\t  Joint 6 (gripper):       B/N\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRecording Controls:\n"
        msg += "\t  Start Recording:         SPACE\n"
        msg += "\t  End Recording (Failed):  P\n"
        msg += "\t  End Recording (Success): M\n"
        msg += "\tControl+C: quit"
        return msg

    def get_device_state(self):
        """Returns the state of both arms."""
        return {
            'left_arm': self._delta_pos_left,
            'right_arm': self._delta_pos_right
        }

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self.started 
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.started
        ac_dict['bi_keyboard'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        return ac_dict

    def reset(self):
        """Reset both arm command buffers."""
        self._delta_pos_left = np.zeros(6)
        self._delta_pos_right = np.zeros(6)

    def add_callback(self, key: str, func: Callable):
        """Add additional callback for specific keys."""
        self._additional_callbacks[key] = func

    def _on_keyboard_event(self, event, *args, **kwargs):
        # Debug: print key events
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            print(f"Key pressed: {event.input.name}")
        
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Left arm controls
            if event.input.name in self._LEFT_ARM_KEY_MAPPING.keys():
                self._delta_pos_left += self._LEFT_ARM_KEY_MAPPING[event.input.name]
            # Right arm controls
            elif event.input.name in self._RIGHT_ARM_KEY_MAPPING.keys():
                self._delta_pos_right += self._RIGHT_ARM_KEY_MAPPING[event.input.name]
            # Recording controls
            elif event.input.name == "SPACE":
                print("SPACE pressed - Starting recording")
                self.started = True
                self._reset_state = False
                # Don't reset position buffers when starting - just set flags
            elif event.input.name == "P":
                print("P pressed - Task failed, stopping recording")
                self.started = False
                self._reset_state = True
                if "P" in self._additional_callbacks:
                    print("Calling P callback")
                    self._additional_callbacks["P"]()
            elif event.input.name == "M":
                print("M pressed - Task success, stopping recording")
                self.started = False
                self._reset_state = True
                if "M" in self._additional_callbacks:
                    print("Calling M callback")
                    self._additional_callbacks["M"]()
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # Left arm controls
            if event.input.name in self._LEFT_ARM_KEY_MAPPING.keys():
                self._delta_pos_left -= self._LEFT_ARM_KEY_MAPPING[event.input.name]
            # Right arm controls
            elif event.input.name in self._RIGHT_ARM_KEY_MAPPING.keys():
                self._delta_pos_right -= self._RIGHT_ARM_KEY_MAPPING[event.input.name]
        return True

    def _create_key_bindings(self):
        """Creates default key binding for both arms."""
        # Left arm key mappings (same as original Se3Keyboard)
        self._LEFT_ARM_KEY_MAPPING = {
            "Q": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "W": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "E": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "A": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.sensitivity,
            "S": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.sensitivity,
            "D": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.sensitivity,
            "U": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "I": np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "O": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "J": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.sensitivity,
            "K": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.sensitivity,
            "L": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.sensitivity,
        }
        
        # Right arm key mappings (new keys)
        self._RIGHT_ARM_KEY_MAPPING = {
            "T": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "R": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "G": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "Z": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.sensitivity,
            "C": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.sensitivity,
            "B": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.sensitivity,
            "Y": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "F": np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "H": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "X": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.sensitivity,
            "V": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.sensitivity,
            "N": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.sensitivity,
        }
