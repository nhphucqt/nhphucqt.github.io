var scrollTopButton = document.getElementById("scrolltop-button");

window.onscroll = function() {scrollFunction();}

function scrollFunction() {
    if (document.documentElement.scrollTop > 100 || document.body.scrollTop > 100) {
        scrollTopButton.style.display = "block";
    }
    else {
        scrollTopButton.style.display = "none";
    }
};

function scrollToTop() {
    document.body.scrollTop = 0;
    document.documentElement.scrollTo({top: 0, behavior: "smooth"});
}

scrollFunction();