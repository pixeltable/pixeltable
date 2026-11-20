#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static long fail_nth = -1;
static long unix_sends = 0;
static int inited = 0;

static void init(void) {
    if (inited) return;
    inited = 1;
    const char *s = getenv("EXP_SEND_FAIL_NTH");
    if (s) fail_nth = atol(s);
}

static int is_unix_socket(int fd) {
    struct sockaddr_storage ss;
    socklen_t len = sizeof(ss);
    if (getsockname(fd, (struct sockaddr *)&ss, &len) != 0) return 0;
    return ss.ss_family == AF_UNIX;
}

static void log_failure(const char *fn, int fd, size_t len, int err) {
    char buf[256];
    time_t t = time(NULL);
    struct tm tm;
    gmtime_r(&t, &tm);
    int n = snprintf(buf, sizeof(buf), "EXP-SEND %02d:%02d:%02d pid %d %s(fd=%d, len=%zu) failed: errno %d (%s) unix=%d\n",
                     tm.tm_hour, tm.tm_min, tm.tm_sec, getpid(), fn, fd, len, err, strerror(err), is_unix_socket(fd));
    const char *path = getenv("EXP_SEND_LOG_PATH");
    int lfd = open(path ? path : "/tmp/exp-sendlog.txt", O_WRONLY | O_APPEND | O_CREAT, 0666);
    if (lfd >= 0) {
        (void)!write(lfd, buf, n);
        close(lfd);
    }
}

static ssize_t my_send(int fd, const void *p, size_t len, int flags) {
    init();
    if (fail_nth >= 0 && is_unix_socket(fd)) {
        if (++unix_sends == fail_nth) {
            log_failure("send[injected]", fd, len, ENOBUFS);
            errno = ENOBUFS;
            return -1;
        }
    }
    ssize_t r = send(fd, p, len, flags);
    if (r < 0 && errno != EAGAIN && errno != EINTR) {
        int e = errno;
        log_failure("send", fd, len, e);
        errno = e;
    }
    return r;
}

static ssize_t my_sendto(int fd, const void *p, size_t len, int flags, const struct sockaddr *a, socklen_t al) {
    ssize_t r = sendto(fd, p, len, flags, a, al);
    if (r < 0 && errno != EAGAIN && errno != EINTR) {
        int e = errno;
        log_failure("sendto", fd, len, e);
        errno = e;
    }
    return r;
}

__attribute__((used)) static struct { const void *replacement; const void *replacee; } interposers[]
    __attribute__((section("__DATA,__interpose"))) = {
        {(const void *)my_send, (const void *)send},
        {(const void *)my_sendto, (const void *)sendto},
};
