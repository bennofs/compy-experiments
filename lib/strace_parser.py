import ast
import os.path
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

SYSCALL_REGEX = re.compile(
    rb'^(?P<pid>[0-9]+) +'
    rb'(?:'
    rb'(?P<syscall>execve|chdir|fchdir|execveat|fork|clone|vfork)'
    rb'\((?P<args>[^)\n]*)'
    rb'(?:\)\s*=\s*(?P<result>[0-9]+)| <unfinished \.\.\.>)$'
    rb'|<\.\.\. (?P<syscall_resumed>fork|clone|vfork) resumed>[^\n]*=\s+(?P<result_resumed>[0-9]+)$'
    rb')',
    re.MULTILINE
)

EXEC_ARGS_REGEX = re.compile(rb'(?P<dirfd>[0-9]+<[^>]+>, )?(?P<exe>"[^"]*"), (?P<argv>\[[^\n]*\]), .*')


@dataclass
class Command:
    workdir: Optional[str]
    exe: str
    argv: List[str]
    pid: int

    __slots__ = ['workdir', 'exe', 'argv', 'pid']

    def to_dict(self):
        return {'workdir': self.workdir, 'exe': self.exe, 'argv': self.argv, 'pid': self.pid}


def decode_fd_arg(arg):
    start = arg.find(b'<')
    end = arg.find(b'>')
    return ast.literal_eval('"' + arg[start+1:end].decode() + '"')


class StraceParser:
    # maps process id to current working directory
    cwd: Dict[int, str]

    # maps process id to queued commands that need to be processed after fork was completed
    queued_matches: Dict[int, List[re.Match]]

    # pid of the overall parent process
    pid: Optional[int]

    def __init__(self, cwd):
        self.cwd = {}
        self.queued_matches = {}
        self.pid = None

        # we don't know our own pid yet, so put this into the 0 key
        # process_buffer will move it to the correct key as soon as we know our pid
        self.cwd[0] = cwd

    def process_buffer(self, buffer, final=False):
        commands = []
        end = buffer.rfind(b'\n') if not final else len(buffer)
        if end == -1:
            return [], buffer

        regex_iter = iter(SYSCALL_REGEX.finditer(buffer[:end]))
        queue = []
        while True:
            was_queued = False
            if queue:
                match = queue.pop(0)
                was_queued = True
            else:
                match = next(regex_iter, None)
                if match is None:
                    if final and self.queued_matches:
                        for matches in self.queued_matches.values():
                            queue.extend(matches)
                        self.queued_matches = {}
                        continue
                    break
            pid = int(match.group('pid'))
            if self.pid is None:
                self.pid = pid
                self.cwd[pid] = self.cwd.pop(0)

            syscall = match.group('syscall') or match.group('syscall_resumed')
            args = match.group('args')

            if syscall == b'fork' or syscall == b'clone' or syscall == b'vfork':
                child_pid = match.group('result') or match.group('result_resumed')
                if child_pid is None:
                    continue
                child_pid = int(child_pid)
                if child_pid != 0:
                    self.cwd[child_pid] = self.cwd.get(pid)
                    queue.extend(self.queued_matches.pop(child_pid, []))

            new_cwd = None
            if syscall == b'chdir':
                new_cwd = ast.literal_eval(args.split(b' ')[0].decode())
            if syscall == b'fchdir':
                new_cwd =  decode_fd_arg(args.split(b' ')[0])
            if new_cwd is not None:
                new_cwd = new_cwd.rstrip('/')
                if not new_cwd.startswith('/'):
                    c = self.cwd.get(pid)
                    if c is None and not was_queued:
                        self.queued_matches.setdefault(pid, []).append(match)
                        continue
                    new_cwd = os.path.normpath(c + "/" + new_cwd)
                self.cwd[pid] = new_cwd
            if syscall == b'execve' or syscall == b'execveat':
                if self.cwd.get(pid) is None and not was_queued:
                    self.queued_matches.setdefault(pid, []).append(match)
                    continue

                match = EXEC_ARGS_REGEX.match(args)

                if not match:
                    print("WARN: cannot parse args", args, file=sys.stderr)
                    continue

                exe = ast.literal_eval(match.group('exe').decode())
                argv = ast.literal_eval(match.group('argv').decode())
                dirfd = match.group('dirfd')
                if dirfd is not None:
                    path = decode_fd_arg(dirfd)
                    exe = os.path.join(path, exe)

                cmd = Command(
                    self.cwd.get(pid),
                    exe,
                    argv,
                    pid,
                )
                commands.append(cmd)

        return commands, buffer[end+1:]


if __name__ == '__main__' and  '__file__' in globals():
    data = sys.stdin.buffer.read()

    import time
    start_time = time.time()
    p = StraceParser("/test")
    p.process_buffer(data)
    end_time = time.time()
    print(len(data) / (end_time - start_time) / 1_000_000, "MB / sec")
