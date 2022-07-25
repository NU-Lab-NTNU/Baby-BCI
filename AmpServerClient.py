import socket
import threading
import time
from collections import deque
import struct
import numpy as np
import logging
import itertools
import traceback
from EEGBuffer import RingBuffer

"""
    Contains two classes for TCP communication between EGI AmpServer and external computer.
    AmpServerSocket contains configuration and socket object for a single port.
    AmpServerClient provides a simple api for interfacing with an operator module of BCI system

    Notes:
        - UI is omitted, might be added at later time.
        - Heavily inspired by labstreaminglayer/App-EGIAmpServer: https://github.com/labstreaminglayer/App-EGIAmpServer

    To do:
        - More comprehensive error handling

    Last edit: 15th of june 2022

    Author: Vegard Kjeka Broen (NTNU)
"""


class PacketFormat1:
    def __init__(self) -> None:
        # Total 1152 bytes
        self.header = None  # uint32_t[8]                           : DINS (Digital Inputs) 1-8/9-16 at bytes 24/25; net type at byte 26.
        self.eeg = None  # float[256], starting at 32nd byte     : EEG data
        self.pib = None  # float[7], starting at 1056th byte     : PIB data
        self.unused1 = None  # float, starting at 1084th byte        : N/A
        self.ref = None  # float, starting at 1088th byte        : Reference channel
        self.com = None  # float, starting at 1092nd byte        : Common channel
        self.unused2 = None  # float, starting at 1096th byte        : N/A
        self.padding = None  # float[13], starting at 1100th byte    : N/A

        self.size = 1152

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

    def read_eeg(self, buf):
        fmt = ">256f"
        return np.array(struct.unpack(fmt, buf[32:1056]))

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

class AmpServerClient:
    def __init__(
        self,
        _sample_rate,
        _n_channels,
        _ringbuffer_time_capacity,
        _socket_address,
        _command_port,
        _notification_port,
        _data_port,
        _amp_id,
        _amp_model,
    ) -> None:

        # In the future config should be fetched from User Interface
        self.sample_rate = _sample_rate
        self.n_channels = _n_channels
        self.amp_addr = _socket_address
        self.command_port = _command_port
        self.notification_port = _notification_port
        self.data_port = _data_port
        self.amp_id = _amp_id
        self.amp_model = _amp_model

        self.scaling_factor = None
        if self.amp_model == "NA300":
            self.scaling_factor = 0.0244140625
        elif self.amp_model == "NA410":
            self.scaling_factor = 0.00009636188
        else:  # NA400
            self.scaling_factor = 0.00009313225

        # Flags
        self.connected = False
        self.stop_flag = False

        # Events
        self.error_encountered = threading.Event()

        # Should probably be its own class
        self.n_samples = self.sample_rate * _ringbuffer_time_capacity
        self.ringbuf = RingBuffer(self.n_samples, self.n_channels)

        # Stream_info
        self.first_packet_received = False
        self.rec_sample_rate = float(self.sample_rate)

        self.data_header = AmpDataPacketHeader()

    def connect(self):
        self.command_socket = AmpServerSocket(
            address=self.amp_addr, port=self.command_port, name="CommandSocket"
        )
        self.notification_socket = AmpServerSocket(
            address=self.amp_addr,
            port=self.notification_port,
            name="NotificationSocket",
        )
        self.data_socket = AmpServerSocket(
            address=self.amp_addr, port=self.data_port, name="DataSocket"
        )

        _, _, _ = self.recvall(verbose=2)
        self.init_amplifier()

    def _send_command(self, command, ampId, channel, value):
        self.command_socket.send_command(command, ampId, channel, value)
        response = self.command_socket.read_chunk(4096, 2)
        return response

    def _send_data_command(self, command, ampId, channel, value):
        self.data_socket.send_command(command, ampId, channel, value)

    def _get_amplifier_details(self):
        response = self._send_command("cmd_GetAmpDetails", str(self.amp_id), "0", "0")
        pretty_response = parse_status_message(repr(response))
        logging.debug(f"AmpDetails\n{pretty_response}")

    def init_amplifier(self):
        logging.info("Initializing amplifier...")

        """ Because it is possible that the amplifier was not properly disconnected from,
        disconnect and shut down before starting. This will ensure that the
        packetCounter is reset. """

        # Stop amp
        stop_response = self._send_command("cmd_Stop", str(self.amp_id), "0", "0")

        # Turn off amp
        set_power_off_response = self._send_command(
            "cmd_SetPower", str(self.amp_id), "0", "0"
        )

        # Set sample rate, hardcoded to 500Hz
        set_sample_rate_response = self._send_command(
            "cmd_SetDecimatedRate", str(self.amp_id), "0", "500"
        )

        # Turn on amp
        set_power_on_response = self._send_command(
            "cmd_SetPower", str(self.amp_id), "0", "1"
        )

        # Start amp
        start_response = self._send_command("cmd_Start", str(self.amp_id), "0", "0")

        """ set to default acquisition mode (note: this should almost surely come before
        the start call...) """
        # Set to default acquisition mode
        set_default_acq_mode_response = self._send_command(
            "cmd_DefaultAcquisitionState", str(self.amp_id), "0", "0"
        )

        logging.info(self.command_socket.name.upper())
        logging.info(f"Stop\n{parse_status_message(repr(stop_response))}")
        logging.info(f"SetPower\n{parse_status_message(repr(set_power_off_response))}")
        logging.info(
            f"SetDecimatedRate\n{parse_status_message(repr(set_sample_rate_response))}"
        )
        logging.info(f"SetPower\n{parse_status_message(repr(set_power_on_response))}")
        logging.info(f"Start\n{parse_status_message(repr(start_response))}")
        logging.info(
            f"DefaultAcquisitionState\n{parse_status_message(repr(set_default_acq_mode_response))}"
        )
        logging.info("Amplifier initialized\n\n")

        self._get_amplifier_details()

    def recvfrom(self, socket, bufsize, verbose=0, parse_data=True):
        chunk = None
        if socket == "command":
            chunk = self.command_socket.read_chunk(bufsize, 0.5)
            name = self.command_socket.name

        elif socket == "notification":
            chunk = self.notification_socket.read_chunk(bufsize, 0.5)
            name = self.notification_socket.name

        elif socket == "data":
            chunk = self.data_socket.read_chunk(bufsize, 1)
            name = self.data_socket.name
            if chunk is not None and parse_data:
                new_bufsize = bufsize - len(chunk)
                if new_bufsize > 0:
                    lst_chunk = list(chunk)
                    while new_bufsize > 0:
                        chunk = self.data_socket.read_chunk(new_bufsize, 1)
                        lst_chunk = lst_chunk + list(chunk)
                        new_bufsize = bufsize - len(lst_chunk)

                    chunk = bytes(lst_chunk)

        else:
            logging.warning(f"Invalid socket name: {socket}")

        if chunk is not None:
            if verbose <= 0:
                """
                Not important
                """
                pass

            elif verbose == 1:
                """
                Minimal
                """
                logging.info(f"{name}: Received {len(chunk)} bytes.\n")

            elif verbose >= 2:
                """
                Status messages, reformat and print.
                """
                logging.debug(
                    f"{name.upper()}\n{parse_status_message(repr(chunk))}\n\n"
                )

        return chunk

    def recvall(self, verbose=1):
        cmd_chunk = self.recvfrom("command", 4096, verbose)
        not_chunk = self.recvfrom("notification", 4096, verbose)
        data_chunk = self.recvfrom("data", 4096, verbose)
        return cmd_chunk, not_chunk, data_chunk

    def stream_good(self):
        # Dummy function at the moment, not sure what to look at
        if self.first_packet_received is False:
            # Stream has not started
            return True

        else:
            if abs(self.rec_sample_rate - self.sample_rate) >= 100:
                return True

            else:
                return True

    def is_ok(self):
        return self.stream_good()

    def close(self):
        self.command_socket.close()
        self.notification_socket.close()
        self.data_socket.close()

    def read_packet_format_1(self):
        """
        Loop for receiving data packets
        TODO:
            -
        """
        counter = 0
        start_time = time.perf_counter()

        try:
            while self.stream_good() and not self.stop_flag:
                # Check sample_rate
                header_buf = None
                while header_buf is None:
                    header_buf = self.recvfrom("data", self.data_header.size)

                self.data_header.read_var(header_buf)

                first_packet = PacketFormat1()
                packet = PacketFormat1()
                n_samples = int(self.data_header.length / first_packet.size)

                if self.data_header.length % first_packet.size:
                    logging.warning(
                        f"data_header.length is not a multiple of packet.size\ndata_header.length: {self.data_header.length}\npacket.size: {first_packet.size}\n"
                    )

                for _ in range(n_samples):
                    # read packet
                    packet_buf = None
                    while packet_buf is None:
                        packet_buf = self.recvfrom("data", first_packet.size)

                    if not self.first_packet_received:
                        first_packet.read_packet(packet_buf)
                        sample = first_packet.eeg
                        self.first_packet_received = True
                    else:
                        sample = packet.read_eeg(packet_buf)

                    # push eeg_data to ring_buffer
                    self.ringbuf.write_sample(sample)

                    counter = counter + 1
                    if counter % 1000 == 0:
                        elapsed_1000 = time.perf_counter() - start_time
                        start_time = time.perf_counter()
                        self.rec_sample_rate = 1000.0 / elapsed_1000
                        logging.info(
                            f"Receiving sample rate: {round(self.rec_sample_rate, 2)} Hz"
                        )

                    if self.stop_flag:
                        break

            self.stop_listening()
            self.stop_flag = False
            self.first_packet_received = False

        except:
            logging.error(
                f"ampclient: Error encountered in read_packet_format_1: {traceback.format_exc()}"
            )
            self.error_encountered.set()

    def get_samples(self, n):
        """
        Get last n samples in ringbuf, scale to get microvolts
        """
        return self.ringbuf.get_samples(n)

    def start_listening(self):
        self._send_data_command("cmd_ListenToAmp", str(self.amp_id), "0", "0")

    def stop_listening(self):
        self._send_data_command("cmd_StopListeningToAmp", str(self.amp_id), "0", "0")

    def set_stop_flag(self):
        self.stop_flag = True


if __name__ == "__main__":
    # logging.basicConfig(filename='ampserverclient.log', filemode='w', level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    amp_client = AmpServerClient()

    amp_client.connect()

    amp_client.start_listening()

    amp_client.read_packet_format_1()

    eeg_arr = np.zeros((amp_client.n_channels, amp_client.n_samples))

    for i in range(amp_client.n_channels):
        for j in range(amp_client.n_samples):
            idx = j * amp_client.n_channels + i
            eeg_arr[i, j] = amp_client.ringbuf[idx]

    np.save("data3s.npy", eeg_arr)

    while 1:
        _ = amp_client.recvfrom("notification", 1024, 2)
        _ = amp_client.recvfrom("command", 1024, 2)
        _ = amp_client.recvfrom("data", 4096, 1, False)
        time.sleep(0.01)
