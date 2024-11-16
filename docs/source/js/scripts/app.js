// Email Form
$('.sign-up-input').focus(function(e) {
  if ($(window).width() > 850) {
    $('.submission-instructions').css('display', 'block');
  }
}).blur(function(e) {
  if ($(window).width() > 850) {
    $('.submission-instructions').css('display', 'none');
  }
});

function sendThankYou() {
  // Hides the email form to show a thank you
  $('.sign-up-container').css('display', 'none');
  $('.thank-you').css('display', 'flex');

  setTimeout(() => {
    // Resets email input to empty string
    $('#mce-EMAIL').val('');
    $('.thank-you').css('display', 'none');
    $( ".sign-up-container" ).fadeIn( "slow");
  }, 8000);
}