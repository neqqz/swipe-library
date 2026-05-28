#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/select.h>

int main(void) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return 1;

    fcntl(sock, F_SETFL, O_NONBLOCK);

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(8080),
        .sin_addr.s_addr = inet_addr("127.0.0.1")
    };

    int ret = connect(sock, (struct sockaddr *)&addr, sizeof(addr));
    if (ret < 0 && errno != EINPROGRESS) {
        close(sock);
        return 1;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(sock, &fds);
    struct timeval tv = {2, 0};

    ret = select(sock + 1, NULL, &fds, NULL, &tv);
    if (ret <= 0) {
        close(sock);
        return 1;
    }

    int so_error;
    socklen_t len = sizeof(so_error);
    getsockopt(sock, SOL_SOCKET, SO_ERROR, &so_error, &len);
    if (so_error) {
        close(sock);
        return 1;
    }

    const char *req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if (send(sock, req, strlen(req), 0) < 0) {
        close(sock);
        return 1;
    }

    FD_ZERO(&fds);
    FD_SET(sock, &fds);
    tv = (struct timeval){2, 0};

    ret = select(sock + 1, &fds, NULL, NULL, &tv);
    if (ret <= 0) {
        close(sock);
        return 1;
    }

    char buf[256];
    ret = recv(sock, buf, sizeof(buf) - 1, 0);
    close(sock);

    if (ret <= 0) return 1;
    buf[ret] = '\0';

    int status = 0;
    if (sscanf(buf, "HTTP/%*s %d", &status) != 1) return 1;

    return status == 200 ? 0 : 1;
}