(function() {
    function displaySearchResults(results, store) {
      var searchResults = document.getElementById('content');
      var appendString = '<h1>Search results for "' + searchTerm + '":</h1><hr class="style-three">';
      if (results.length) { // Are there any results?
        for (var i = 0; i < results.length; i++) {  // Iterate over the results
          var item = store[results[i].ref];
          appendString += '<a href="' + item.url + '"><h2>' + item.title + '</h2></a>';
          appendString += '<p>' + item.content.substring(0, 150) + '...</p>';
          appendString += '<hr class="style-six">';
        }
      } else {
        appendString += '<p>No results found</p>';
      }
      searchResults.innerHTML = appendString;
    }
  
    function getQueryVariable(variable) {
      var query = window.location.search.substring(1);
      var vars = query.split('&');
  
      for (var i = 0; i < vars.length; i++) {
        var pair = vars[i].split('=');
  
        if (pair[0] === variable) {
          return decodeURIComponent(pair[1].replace(/\+/g, '%20'));
        }
      }
    }
  
    var searchTerm = getQueryVariable('query');
  
    if (searchTerm) {
      document.getElementById('search-box').setAttribute("value", searchTerm);
  
      // Initalize lunr with the fields it will be searching on. I've given title
      // a boost of 10 to indicate matches on this field are more important.
      var idx = lunr(function () {
        this.field('id');
        this.field('title', { boost: 10 });
        this.field('author');
        this.field('category');
        this.field('content');
        this.field('keywords');
      });
  
      for (var key in window.store) { // Add the data to lunr
        idx.add({
          'id': key,
          'title': window.store[key].title,
          'author': window.store[key].author,
          'category': window.store[key].category,
          'content': window.store[key].content,
          'keywords': window.store[key].keywords
        });
  
        var results = idx.search(searchTerm); // Get lunr to perform a search
        displaySearchResults(results, window.store); // We'll write this in the next section
      }
    }
  })();