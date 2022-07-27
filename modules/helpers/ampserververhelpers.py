import struct
import numpy as np
import socket
import enum

class NetCode(enum.Enum):
    GSN64_2_0 = 0       #  GSN 64
    GSN128_2_0 = 1      #  GSN 128
    GSN256_2_0 = 2      #  GSN 256

    HCGSN32_1_0 = 3     #  HGSN 32
    HCGSN64_1_0 = 4     #  HGSN 64
    HCGSN128_1_0 = 5    #  HGSN 128
    HCGSN256_1_0 = 6    #  HGSN 256

    MCGSN32_1_0 = 7     #  MGSN 32
    MCGSN64_1_0 = 8     #  MGSN 64
    MCGSN128_1_0 = 9    #  MGSN 128
    MCGSN256_1_0 = 10   #  MGSN 256

    AMP_SAMPLE = 11     # EAmpSample displayable channels (internal use only)
    TestConnector = 14
    NoNet = 15          # net not connected
    Unknown = 255       # Unknown or net not connected


class PacketFormat1:
    def __init__(self) -> None:
        # Total 1152 bytes
        self.header = None  # uint32_t[8]                           : DINS (Digital Inputs) 1-8/9-16 at bytes 24/25; net type at byte 26.
        self.eeg = None  # float[256], starting at 32nd byte        : EEG data
        self.pib = None  # float[7], starting at 1056th byte        : PIB data
        self.unused1 = None  # float, starting at 1084th byte       : N/A
        self.ref = None  # float, starting at 1088th byte           : Reference channel
        self.com = None  # float, starting at 1092nd byte           : Common channel
        self.unused2 = None  # float, starting at 1096th byte       : N/A
        self.padding = None  # float[13], starting at 1100th byte   : N/A

        self.size = 1152
        self.net_codes = NetCode
        self.net_code_val = None

    def read_packet(self, buf):
        fmt = ">8I"
        self.header = struct.unpack(fmt, buf[0:32])
        fmt = ">256f"
        self.eeg = np.array(struct.unpack(fmt, buf[32:1056]))
        fmt = ">7f"
        self.pib = struct.unpack(fmt, buf[1056:1084])
        fmt = ">f"
        self.unused1 = struct.unpack(fmt, buf[1084:1088])
        self.ref = struct.unpack(fmt, buf[1088:1092])
        self.com = struct.unpack(fmt, buf[1092:1096])
        self.unused2 = struct.unpack(fmt, buf[1096:1100])
        fmt = ">13f"
        self.padding = struct.unpack(fmt, buf[1100:])
        self.net_code_val = int(buf[25])

    def read_eeg(self, buf):
        fmt = ">256f"
        return np.array(struct.unpack(fmt, buf[32:1056]))

    def get_net_code(self):
        net_code = self.net_codes(self.net_code_val)
        n_channels = 0
        if (net_code == self.net_codes.HCGSN32_1_0 or net_code == self.net_codes.MCGSN32_1_0):
            n_channels = 32
        elif (net_code == self.net_codes.GSN64_2_0 or net_code == self.net_codes.HCGSN64_1_0 or net_code == self.net_codes.MCGSN64_1_0):
            n_channels = 64
        elif (net_code == self.net_codes.GSN128_2_0 or net_code == self.net_codes.HCGSN128_1_0 or net_code == self.net_codes.MCGSN128_1_0):
            n_channels = 128
        elif (net_code == self.net_codes.GSN256_2_0 or net_code == self.net_codes.HCGSN256_1_0 or net_code == self.net_codes.MCGSN256_1_0):
            n_channels = 256

        return net_code, n_channels

class AmpDataPacketHeader:
    def __init__(self) -> None:
        # Total 16 bytes
        self.amp_id = None  # int64_t
        self.length = None  # uint64_t, starting at 8th byte
        self.size = 16

    def read_amp_id(self, buf):
        self.amp_id = int.from_bytes(buf, "big")

    def read_length(self, buf):
        self.length = int.from_bytes(buf, "big")

    def read_var(self, buf):
        self.read_amp_id(buf[0:8])
        self.read_length(buf[8:])

def parse_status_message(msg, start_indent=-1):
    """
    Input assumed to be string, not byte object
     *      status				: complete
             *		serial_number		: A14100120
             *		amp_type			: NA400
             *		legacy_board		: false
             *		packet_format		: 2
             *		system_version		: 2.0.14
             *		number_of_channels	: 32
    """

    msg = msg[2:-1]
    ret_msg = ""
    indent_level = start_indent
    last_par = ""
    for i, c in enumerate(msg):
        if c == "(":
            indent_level = indent_level + 1
            if i > 0:
                if last_par == "(":
                    ret_msg = ret_msg + ":"

                ret_msg = ret_msg + "\n" + "\t" * indent_level

            last_par = "("

        elif c == ")":
            indent_level = indent_level - 1
            last_par = ")"

        elif (c == "\\" and msg[i + 1] == "n") or (c == "n" and msg[i - 1] == "\\"):
            continue

        elif c == " " and msg[i + 1] == "(":
            continue

        else:
            ret_msg = ret_msg + c

    return ret_msg + "\n"

class AmpServerSocket:
    def __init__(self, address, port, name) -> None:
        self.name = name
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((address, port))

    def read_chunk(self, bufsize, timeout):
        self.s.settimeout(timeout)
        chunk = None
        try:
            chunk = self.s.recv(bufsize)
        except socket.timeout:
            pass

        self.s.settimeout(None)
        return chunk

    def send_command(self, command, ampId, channel, value):
        str_msg = (
            "(sendCommand "
            + command
            + " "
            + ampId
            + " "
            + channel
            + " "
            + value
            + ")\n"
        )
        byte_msg = str_msg.encode("utf-8")

        self.s.settimeout(1)
        try:
            self.s.sendall(byte_msg)
        except socket.timeout:
            pass

        self.s.settimeout(None)

    def close(self):
        self.s.shutdown()
        self.s.close()