import logging
import os
import time

class Logger:

    def __init__(self, logdir, rank, type='torch', debug=False, filename=None, summary=True, step=None):
        self.logger = None
        self.type = type
        self.rank = rank
        self.step = step
        self.logdir_results = os.path.join(os.path.dirname(os.path.dirname(logdir)), "results")
        self.summary = summary

        self.file_logger = self.setup_logger(logdir, filename)

        if summary:
            if type == 'tensorboardX':
                import tensorboardX
                self.logger = tensorboardX.SummaryWriter(logdir)
            elif type == "torch":
                from torch.utils.tensorboard import SummaryWriter
                self.logger = SummaryWriter(logdir)
            else:
                raise NotImplementedError
        else:
            self.type = 'None'

        self.debug_flag = debug
        # logging.basicConfig(filename=filename, level=logging.INFO, format=f'%(levelname)s:rank{rank}: %(message)s')

        if rank == 0:
            os.makedirs(self.logdir_results, exist_ok=True)
            logging.info(f"[!] starting logging at directory {logdir}")
            if self.debug_flag:
                logging.info(f"[!] Entering DEBUG mode")

    def close(self):
        if self.logger is not None:
            self.logger.close()
        self.info("Closing the Logger.")

    def add_scalar(self, tag, scalar_value, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.logger.add_scalar(tag, scalar_value, step)

    def add_image(self, tag, image, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.logger.add_image(tag, image, step)

    def add_figure(self, tag, image, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.logger.add_figure(tag, image, step)

    def add_table(self, tag, tbl, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            tbl_str = "<table width=\"100%\"> "
            tbl_str += "<tr> \
                     <th>Term</th> \
                     <th>Value</th> \
                     </tr>"
            for k, v in tbl.items():
                tbl_str += "<tr> \
                           <td>%s</td> \
                           <td>%s</td> \
                           </tr>" % (k, v)

            tbl_str += "</table>"
            self.logger.add_text(tag, tbl_str, step)

    def add_results(self, results, tag="Results"):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            text = "<table width=\"100%\">"
            for k, res in results.items():
                text += f"<tr><td>{k}</td>" + " ".join([str(f'<td>{x}</td>') for x in res.values()]) + "</tr>"
            text += "</table>"
            self.logger.add_text(tag, text)

    def print(self, msg):
        self.file_logger.info(msg, extra={'rank': self.rank})

    def info(self, msg):
        self.file_logger.info(msg, extra={'rank': self.rank})
        if self.rank == 0:
            logging.info(msg)

    def debug(self, msg):
        if self.debug_flag:
            self.file_logger.debug(msg, extra={'rank': self.rank})
            if self.rank == 0:
                logging.debug(msg)

    def error(self, msg):
        self.file_logger.error(msg, extra={'rank': self.rank})
        if self.rank == 0:
            logging.error(f"[Rank {self.rank}] {msg}")

    def log_results(self, task, name, results, novel=False):
        if self.rank == 0:
            file_name = f"{task.task}-n{task.nshot}.csv" if task.nshot != -1 else f"{task.task}-0"
            file_name = file_name if not novel else f"{file_name}_novel.csv"
            dir_path = f"{self.logdir_results}/{task.dataset}"
            path = f"{dir_path}/{file_name}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            text = [str(round(time.time())), name, str(self.step), str(task.nshot), str(task.ishot)]
            for val in results:
                text.append(str(val))
            row = ",".join(text) + "\n"
            with open(path, "a") as file:
                file.write(row)

    def log_aggregates(self, task, name, results):
        if self.rank == 0:
            file_name = f"{task.task}-n{task.nshot}-agg.csv" if task.nshot != -1 else f"{task.task}-0-agg.csv"
            # file_name = f"{task.task}-agg.csv" if task.nshot != -1 else f"{task.task}-0-agg.csv"
            dir_path = f"{self.logdir_results}/{task.dataset}"
            path = f"{dir_path}/{file_name}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            text = [str(round(time.time())), name, str(self.step), str(task.nshot), str(task.ishot)]
            for val in results:
                text.append(str(val))
            row = ",".join(text) + "\n"
            # # 检查文件是否已存在
            # file_exists = os.path.exists(path)

            with open(path, "a") as file:
                # 如果文件已存在且不为空，则在写入新数据前添加空行
                # if file_exists and os.path.getsize(path) > 0:
                #     file.write("\n\n\n\n\n")  # 写入多个空行
                file.write(row)

    def _transform_tag(self, tag):
        tag = tag + f"/{self.step}" if self.step is not None else tag
        return tag

    def is_not_none(self):
        return self.type != "None"

    def setup_logger(self, logdir, filename):
        log_dir = os.path.join(os.path.dirname(os.path.dirname(logdir)), "log/")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{filename}.log")

        formatter = logging.Formatter('%(asctime)s [Rank %(rank)s] %(message)s')
        logger = logging.getLogger(f"file_logger_{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if logger.handlers:
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if self.rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger