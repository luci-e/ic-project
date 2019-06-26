if (window.addthis_product === undefined) { window.addthis_product = "wpwt"; } if (window.wp_product_version === undefined) { window.wp_product_version = "wpwt-3.1.0"; } if (window.wp_blog_version === undefined) { window.wp_blog_version = "4.8.7"; } if (window.addthis_share === undefined) { window.addthis_share = {}; } if (window.addthis_config === undefined) { window.addthis_config = {"data_track_clickback":true,"ui_atversion":"300"}; } if (window.addthis_plugin_info === undefined) { window.addthis_plugin_info = {"info_status":"enabled","cms_name":"WordPress","plugin_name":"Website Tools by AddThis","plugin_version":"3.1.0","plugin_mode":"AddThis","anonymous_profile_id":"wp-6f77069998c0259a251c5cd50561455e","page_info":{"template":false}}; } 
                    (function() {
                      var first_load_interval_id = setInterval(function () {
                        if (typeof window.addthis !== 'undefined') {
                          window.clearInterval(first_load_interval_id);
                          if (typeof window.addthis_layers !== 'undefined' && Object.getOwnPropertyNames(window.addthis_layers).length > 0) {
                            window.addthis.layers(window.addthis_layers);
                          }
                          if (Array.isArray(window.addthis_layers_tools)) {
                            for (i = 0; i < window.addthis_layers_tools.length; i++) {
                              window.addthis.layers(window.addthis_layers_tools[i]);
                            }
                          }
                        }
                     },1000)
                    }());
                