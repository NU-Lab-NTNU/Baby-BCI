import time
import numpy as np
import logging
import traceback

from modules.helpers.EEGBuffer import RingBuffer
import modules.helpers.ampserververhelpers as amp
from modules.SubModule import SubModule


class AmpServerClient(SubModule):
    def __init__(
        self,
        _sample_rate,
        _n_channels,
        _mode,
        _ringbuffer_time_capacity,
        _socket_address,
        _command_port,
        _notification_port,
        _data_port,
        _amp_id,
        _amp_model,
    ) -> None:
        # Initialize parent class
        super().__init__()

        # In the future config should be fetched from User Interface
        self.sample_rate = _sample_rate
        self.n_channels = _n_channels
        self.mode = _mode
        self.amp_addr = _socket_address
        self.command_port = _command_port
        self.notification_port = _notification_port
        self.data_port = _data_port
        self.amp_id = _amp_id
        self.amp_model = _amp_model

        # Scaling factor
        self.scaling_factor = None
        if self.amp_model == "NA300":
            self.scaling_factor = 0.0244140625
        elif self.amp_model == "NA410":
            self.scaling_factor = 0.00009636188
        else:  # NA400
            self.scaling_factor = 0.00009313225

        # EEG ringbuffer
        self.n_samples = self.sample_rate * _ringbuffer_time_capacity
        self.ringbuf = RingBuffer(self.n_samples, self.n_channels)

        # Stream_info
        self.first_packet_received = False
        self.rec_sample_rate = float(self.sample_rate)

        # Packets
        self.data_header = amp.AmpDataPacketHeader()

        # Duplication factor
        self.duplication_factor = int(1000 / self.sample_rate)

        # Assert config is good
        self.assert_config()

    def assert_config(self):
        # Assert correct sample_rate
        if self.amp_model == "NA300":
            allowed_fs = [50, 100, 200, 250, 500, 1000]
        else:
            allowed_fs = [250, 500, 1000]

        assert isinstance(self.sample_rate, int)
        assert self.sample_rate in allowed_fs

        # Assert correct n_channels
        allowed_n_channels = [32, 64, 128, 256]
        assert isinstance(self.n_channels, int)
        assert self.n_channels in allowed_n_channels

        # Assert duplication factor
        assert isinstance(self.duplication_factor, int)
        assert self.duplication_factor * self.sample_rate == 1000

    def connect(self):
        try:
            self.command_socket = amp.AmpServerSocket(
                address=self.amp_addr, port=self.command_port, name="CommandSocket"
            )
            self.notification_socket = amp.AmpServerSocket(
                address=self.amp_addr,
                port=self.notification_port,
                name="NotificationSocket",
            )
            self.data_socket = amp.AmpServerSocket(
                address=self.amp_addr, port=self.data_port, name="DataSocket"
            )

            _, _, _ = self.recvall(verbose=2)
            self.init_amplifier()
        except:
            logging.error(
                f"ampclient: Error encountered in connect: {traceback.format_exc()}"
            )
            self.set_error_encountered()

    def send_cmd(self, command, ampId, channel, value):
        self.command_socket.send_command(command, ampId, channel, value)
        response = self.command_socket.read_chunk(4096, 2)
        return response

    def send_data_cmd(self, command, ampId, channel, value):
        self.data_socket.send_command(command, ampId, channel, value)

    def _get_amplifier_details(self):
        response = self.send_cmd("cmd_GetAmpDetails", str(self.amp_id), "0", "0")
        pretty_response = amp.parse_status_message(repr(response))
        logging.debug(f"AmpDetails\n{pretty_response}")

    def init_amplifier(self):
        logging.info("Initializing amplifier...")

        """ Because it is possible that the amplifier was not properly disconnected from,
        disconnect and shut down before starting. This will ensure that the
        packetCounter is reset. """

        set_mode_response = self.send_cmd(
            "cmd_DefaultAcquisitionState", str(self.amp_id), "0", "0"
        )
        mode = "DefaultAcquisitionState"

        # =============================================================================
        #         # Stop amp
        #         stop_response = self.send_cmd("cmd_Stop", str(self.amp_id), "0", "0")
        #
        #         # Turn off amp
        #         set_power_off_response = self.send_cmd(
        #             "cmd_SetPower", str(self.amp_id), "0", "0"
        #         )
        #
        #         # Turn on amp
        #         set_power_on_response = self.send_cmd(
        #             "cmd_SetPower", str(self.amp_id), "0", "1"
        #         )
        # =============================================================================

        # Turn on Filter and Decimation routines
        set_filter_and_decimate_response = self.send_cmd(
            "cmd_SetFilterAndDecimate", str(self.amp_id), "0", "1"
        )

        # Set sample rate
        fs = str(self.sample_rate)
        set_sample_rate_response = self.send_cmd(
            "cmd_SetDecimatedRate", str(self.amp_id), "0", fs
        )

        """ set to default acquisition or default signal generation mode, depending on config (note: this should almost surely come before
        the start call...) """
        if self.mode == "test":
            set_mode_response = self.send_cmd(
                "cmd_DefaultSignalGeneration", str(self.amp_id), "0", "0"
            )
            mode = "DefaultSignalGeneration"
        else:
            set_mode_response = self.send_cmd(
                "cmd_DefaultAcquisitionState", str(self.amp_id), "0", "0"
            )
            mode = "DefaultAcquisitionState"

        # Start data stream
        start_response = self.send_cmd("cmd_Start", str(self.amp_id), "0", "0")

        logging.debug(self.command_socket.name.upper())
        # logging.debug(f"Stop\n{amp.parse_status_message(repr(stop_response))}")
        # logging.debug(f"SetPower\n{amp.parse_status_message(repr(set_power_off_response))}")
        # logging.debug(f"SetPower\n{amp.parse_status_message(repr(set_power_on_response))}")
        logging.debug(
            f"SetFilterAndDecimate\n{amp.parse_status_message(repr(set_filter_and_decimate_response))}"
        )
        logging.debug(
            f"SetDecimatedRate\n{amp.parse_status_message(repr(set_sample_rate_response))}"
        )
        logging.debug(f"{mode}\n{amp.parse_status_message(repr(set_mode_response))}")
        logging.debug(f"Start\n{amp.parse_status_message(repr(start_response))}")
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
                    f"{name.upper()}\n{amp.parse_status_message(repr(chunk))}\n\n"
                )

        return chunk

    def recvall(self, verbose=1):
        cmd_chunk = self.recvfrom("command", 4096, verbose)
        not_chunk = self.recvfrom("notification", 4096, verbose)
        data_chunk = self.recvfrom("data", 4096, verbose)
        return cmd_chunk, not_chunk, data_chunk

    def is_duplicate(self, counter):
        return counter % self.duplication_factor

    def main_loop(self):
        """
        Loop for receiving data packets
        TODO:
            - enable reading of packet format 2
        """
        counter = 0
        unique_counter = 0
        start_time = time.perf_counter()

        try:
            while self.is_ok():
                # Check sample_rate
                header_buf = None
                while header_buf is None:
                    header_buf = self.recvfrom("data", self.data_header.size)

                self.data_header.read_var(header_buf)

                first_packet = amp.PacketFormat1()
                packet = amp.PacketFormat1()
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
                        net_code, n_channels = first_packet.get_net_code()
                        logging.info(
                            f"ampclient: net_code = {net_code.name}, n_channels = {n_channels}"
                        )
                        self.first_packet_received = True
                    else:
                        sample = packet.read_eeg(packet_buf)

                    # push eeg_data to ring_buffer
                    if not self.is_duplicate(counter):
                        self.ringbuf.write_sample(sample)
                        unique_counter = unique_counter + 1

                        if unique_counter % 1000 == 0:
                            elapsed_1000 = time.perf_counter() - start_time
                            start_time = time.perf_counter()
                            self.rec_sample_rate = 1000.0 / elapsed_1000
                            logging.info(
                                f"ampclient: unique data packet rate: {round(self.rec_sample_rate, 2)} Hz"
                            )

                    counter = counter + 1

                    if not self.is_ok():
                        break

            self.first_packet_received = False

        except:
            logging.error(
                f"ampclient: Error encountered in main_loop: {traceback.format_exc()}"
            )
            self.set_error_encountered()

        logging.info("ampclient: exiting main_loop")
        self.close()
        self.stop_flag = False

    def get_samples(self, n):
        """
        Get last n samples in ringbuf, scale to get microvolts
        """
        x, read_time = self.ringbuf.get_samples(n)
        return x * self.scaling_factor, read_time

    def start_listening(self):
        self.send_data_cmd("cmd_ListenToAmp", str(self.amp_id), "0", "0")

    def stop_amp(self):
        self.send_data_cmd("cmd_StopListeningToAmp", str(self.amp_id), "0", "0")
        _ = self.send_cmd("cmd_Stop", str(self.amp_id), "0", "0")

    def close(self):
        self.stop_amp()

        self.command_socket.close()
        self.notification_socket.close()
        self.data_socket.close()

if __name__ == "__main__":
    # logging.basicConfig(filename='ampserverclient.log', filemode='w', level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    amp_client = AmpServerClient()

    amp_client.connect()

    amp_client.start_listening()

    amp_client.main_loop()

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
